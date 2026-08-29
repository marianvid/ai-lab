"""The download queue, and moving the bytes.

One background worker at a time, on purpose: several parallel downloads of
multi-gigabyte files compete for the same disk and finish no sooner, while
making progress harder to read.

Three habits keep an interrupted transfer from turning into a broken model.

A model is assembled out of sight. Everything is written into a working folder
whose name begins with a dot, beside the repository it is bound for, and the
library does not look inside folders named that way. Only when every file has
arrived and been checked is the folder moved into place, which on one
filesystem is a single rename: either the model is there or it is not, and
there is no moment where it is half there under its real name. This matters
most for a model made of several files, where the old behaviour would have
shown the two that had arrived as models of their own while the third was
still coming.

Every file is checked before that move. The size must match the listing, and
where Hugging Face publishes a SHA-256 — which it does for every large file —
the bytes on disk are hashed and compared. A truncated download that ends in a
clean-looking file is caught here rather than by an engine failing to load it
a week later.

A restarted transfer resumes rather than starting over. Each file is written
to `<name>.part` and renamed within the working folder once complete, and the
working folder is named after the transfer, so a cancelled 20 GB download
continues from where it stopped instead of beginning again.
"""

from __future__ import annotations

import os
import shutil
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from queue import Queue
from typing import Callable

from ..events import EventBus
from ..types import ChangeEvent
from .huggingface import RemoteFile, RemoteSet

CHUNK = 1024 * 1024
USER_AGENT = "AI-Lab/0.2"
# Where a model is assembled before it is published. It sits beside the
# repository the model is bound for, so the move into place is a rename on one
# filesystem rather than a copy across two. The leading dot is what keeps it
# out of the library: `Catalog` does not look inside folders named this way.
WORKING = ".ai-lab-downloads"
# How often a running transfer says it has moved. Announcing every chunk would
# redraw the page dozens of times a second to show a number that changes by a
# percent.
ANNOUNCE_EVERY_S = 0.5


class State(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(slots=True)
class Transfer:
    """One model on its way to disk."""

    id: str
    repo: str
    name: str
    destination: str
    total_bytes: int
    state: State = State.QUEUED
    received_bytes: int = 0
    files_done: int = 0
    files_total: int = 0
    error: str = ""
    storage_tier: str = "core"
    # What the check before publication actually proved, so a report can say
    # so instead of implying every file was hashed.
    checked_hash: int = 0
    checked_size: int = 0
    _cancel: threading.Event = field(default_factory=threading.Event, repr=False)

    @property
    def percent(self) -> float:
        if self.total_bytes <= 0:
            return 0.0
        return min(100.0, self.received_bytes * 100.0 / self.total_bytes)

    def json(self) -> dict:
        return {"id": self.id, "repo": self.repo, "name": self.name,
                "state": self.state.value, "percent": round(self.percent, 1),
                "received_bytes": self.received_bytes,
                "total_bytes": self.total_bytes,
                "files_done": self.files_done, "files_total": self.files_total,
                "error": self.error, "storage_tier": self.storage_tier,
                "checked_hash": self.checked_hash,
                "checked_size": self.checked_size}


class DownloadManager:
    """Accepts model sets, downloads them one at a time, reports progress."""

    def __init__(self, opener: Callable | None = None,
                 bus: EventBus | None = None,
                 arrived: Callable | None = None) -> None:
        self._bus = bus
        # Run once when a model has finished arriving, on the worker thread.
        # This module does not know or care what the work is; whoever built it
        # decided that. Today it is reading the new files to find out what the
        # model can do, which takes a quarter of a second and is much better
        # spent here than in front of somebody opening a page.
        self._arrived = arrived
        self._announced = 0.0
        self._transfers: dict[str, Transfer] = {}
        self._plans: dict[str, tuple[RemoteSet, Path]] = {}
        self._queue: Queue[str] = Queue()
        self._lock = threading.RLock()
        self._open = opener or _open_range
        self._worker: threading.Thread | None = None

    # -- queue -------------------------------------------------------------

    def enqueue(self, remote: RemoteSet, destination: Path,
                storage_tier: str = "core") -> Transfer:
        """Queue a whole model. Never a single file — see the package docstring."""
        if not remote.complete:
            raise ValueError(
                f"{remote.name} is missing {len(remote.missing)} shard(s) upstream")
        transfer = Transfer(
            id=f"{remote.repo}/{remote.name}".replace("/", "_"),
            repo=remote.repo, name=remote.name, destination=str(destination),
            total_bytes=remote.size_bytes, files_total=len(remote.files),
            storage_tier=storage_tier,
        )
        with self._lock:
            if transfer.id in self._transfers and \
                    self._transfers[transfer.id].state in (State.QUEUED, State.RUNNING):
                raise ValueError(f"{remote.name} is already downloading")
            self._transfers[transfer.id] = transfer
            self._plans[transfer.id] = (remote, destination)
        self._queue.put(transfer.id)
        self._ensure_worker()
        return transfer

    def cancel(self, transfer_id: str) -> None:
        with self._lock:
            transfer = self._transfers.get(transfer_id)
        if transfer is None:
            raise KeyError(transfer_id)
        transfer._cancel.set()
        if transfer.state in (State.QUEUED, State.RUNNING):
            transfer.state = State.CANCELLED
            self._announce(force=True)

    def list(self) -> list[dict]:
        with self._lock:
            return [item.json() for item in self._transfers.values()]

    # -- the worker --------------------------------------------------------

    def _ensure_worker(self) -> None:
        with self._lock:
            if self._worker and self._worker.is_alive():
                return
            self._worker = threading.Thread(target=self._run, daemon=True,
                                            name="ai-lab-downloads")
            self._worker.start()

    def _run(self) -> None:
        while True:
            try:
                transfer_id = self._queue.get(timeout=2.0)
            except Exception:
                return
            transfer = self._transfers.get(transfer_id)
            plan = self._plans.get(transfer_id)
            if transfer is None or plan is None or transfer.state is State.CANCELLED:
                continue
            self._download(transfer, *plan)

    def _announce(self, force: bool = False) -> None:
        if self._bus is None:
            return
        now = time.monotonic()
        if force or now - self._announced >= ANNOUNCE_EVERY_S:
            self._announced = now
            self._bus.publish(ChangeEvent(topic="downloads"))

    def _download(self, transfer: Transfer, remote: RemoteSet, destination: Path) -> None:
        transfer.state = State.RUNNING
        self._announce(force=True)
        working = _working(destination, transfer.id)
        try:
            working.mkdir(parents=True, exist_ok=True)
            os.chmod(working, 0o700)
            staged = []
            for item in remote.files:
                if transfer._cancel.is_set():
                    transfer.state = State.CANCELLED
                    return
                # A file already sitting at its final name, the right size, is
                # one somebody has already waited for. Asking for the same
                # model again should not fetch twenty gigabytes a second time.
                if _already_there(destination, item):
                    transfer.received_bytes += item.size_bytes
                    transfer.files_done += 1
                    continue
                self._file(transfer, remote.repo, item, working)
                staged.append(item)
                transfer.files_done += 1
            _verify(transfer, remote, working, destination, staged)
            _publish(working, destination, staged)
            transfer.state = State.DONE
        except Exception as error:
            if transfer._cancel.is_set():
                transfer.state = State.CANCELLED
                transfer.error = ""
            else:
                transfer.state = State.FAILED
                transfer.error = str(error) or error.__class__.__name__
        # The working folder is deliberately left behind on anything but
        # success. It holds the bytes already fetched, and asking for the same
        # model again continues from them.
        # A finished transfer also means a new model on disk.
        self._announce(force=True)
        if transfer.state is State.DONE:
            if self._bus is not None:
                self._bus.publish(ChangeEvent(topic="models"))
            if self._arrived is not None:
                # Never let this stop the queue. The download succeeded; that
                # is the promise made to whoever asked for it, and whatever
                # this was going to do can be done again later.
                try:
                    self._arrived(destination)
                except Exception:
                    pass

    def _file(self, transfer: Transfer, repo: str, item: RemoteFile,
              destination: Path) -> None:
        """Fetch one file into the working folder.

        Every file lands flat, under its own name and not the folders it sat
        in upstream. That is what a ComfyUI model needs — it looks a part up by
        file name in the directory it is pointed at — and it is why two parts
        of one bundle may not share a name.
        """
        target = destination / Path(item.path).name
        if target.exists() and target.stat().st_size == item.size_bytes:
            transfer.received_bytes += item.size_bytes
            return
        partial = target.with_name(target.name + ".part")
        resume_from = partial.stat().st_size if partial.exists() else 0
        transfer.received_bytes += resume_from

        with self._open(item.url(repo), resume_from) as response:
            with open(partial, "ab" if resume_from else "wb") as handle:
                while True:
                    if transfer._cancel.is_set():
                        raise RuntimeError("cancelled")
                    chunk = response.read(CHUNK)
                    if not chunk:
                        break
                    handle.write(chunk)
                    transfer.received_bytes += len(chunk)
                    self._announce()
        partial.replace(target)


def _open_range(url: str, resume_from: int):
    headers = {"User-Agent": USER_AGENT}
    if resume_from:
        headers["Range"] = f"bytes={resume_from}-"
    request = urllib.request.Request(url, headers=headers)
    try:
        return urllib.request.urlopen(request, timeout=60)
    except urllib.error.HTTPError as error:
        raise ValueError(f"Download failed with HTTP {error.code}") from None
    except OSError as error:
        raise ValueError(f"Download failed: {error}") from None


def _working(destination: Path, transfer_id: str) -> Path:
    """The folder a model is assembled in, beside the repository it belongs to."""
    return destination.parent / WORKING / _safe(transfer_id)


def _safe(name: str) -> str:
    """A transfer id reduced to something that is only ever a folder name."""
    return "".join(character if character.isalnum() or character in "-._"
                   else "_" for character in name).lstrip(".") or "transfer"


def _already_there(destination: Path, item: RemoteFile) -> bool:
    final = destination / Path(item.path).name
    return bool(item.size_bytes) and final.is_file() \
        and final.stat().st_size == item.size_bytes


def _verify(transfer: Transfer, remote: RemoteSet, working: Path,
            destination: Path, staged: list) -> None:
    """Check every file before any of it becomes visible.

    Size is checked always. The hash is checked wherever upstream published
    one, and how many files got which check is recorded on the transfer, so
    nothing here has to be taken on trust afterwards.

    Files that were already in place are checked where they are. They are part
    of the model about to be published, so saying nothing about them would
    make the count above a half-truth.
    """
    transfer.checked_hash = transfer.checked_size = 0
    for item in remote.files:
        folder = working if item in staged else destination
        path = folder / Path(item.path).name
        if not path.is_file():
            raise ValueError(f"{path.name} did not arrive")
        actual = path.stat().st_size
        if item.size_bytes and actual != item.size_bytes:
            raise ValueError(
                f"{path.name} is {actual} bytes; upstream says {item.size_bytes}")
        transfer.checked_size += 1
        if not item.sha256:
            continue
        if _sha256(path) != item.sha256:
            raise ValueError(
                f"{path.name} arrived complete but its contents do not match "
                "the checksum upstream publishes for it")
        transfer.checked_hash += 1


def _sha256(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _publish(working: Path, destination: Path, staged: list) -> None:
    """Make the finished model visible, in one act where the filesystem allows.

    Nothing has been downloaded to this name before, which is the ordinary
    case, so the whole working folder is renamed into place and the model
    appears complete or not at all. When something is already there — the same
    model fetched again, or a repository root that holds loose files — the
    files are moved one at a time into it, which is the best a filesystem
    offers for merging into a directory that exists.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        os.replace(working, destination)
        return
    for item in staged:
        name = Path(item.path).name
        os.replace(working / name, destination / name)
    shutil.rmtree(working, ignore_errors=True)
