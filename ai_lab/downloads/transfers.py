"""The download queue, and moving the bytes.

One background worker at a time, on purpose: several parallel downloads of
multi-gigabyte files compete for the same disk and finish no sooner, while
making progress harder to read.

Two habits keep an interrupted transfer from turning into a broken model.
Files are written to `<name>.part` and renamed only once complete, so the
catalog never sees a half-written file and calls it a model. And a restarted
transfer resumes with a Range request instead of starting over, because losing
20 GB to a dropped connection is not acceptable.
"""

from __future__ import annotations

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
                "error": self.error}


class DownloadManager:
    """Accepts model sets, downloads them one at a time, reports progress."""

    def __init__(self, opener: Callable | None = None,
                 bus: EventBus | None = None) -> None:
        self._bus = bus
        self._announced = 0.0
        self._transfers: dict[str, Transfer] = {}
        self._plans: dict[str, tuple[RemoteSet, Path]] = {}
        self._queue: Queue[str] = Queue()
        self._lock = threading.RLock()
        self._open = opener or _open_range
        self._worker: threading.Thread | None = None

    # -- queue -------------------------------------------------------------

    def enqueue(self, remote: RemoteSet, destination: Path) -> Transfer:
        """Queue a whole model. Never a single file — see the package docstring."""
        if not remote.complete:
            raise ValueError(
                f"{remote.name} is missing {len(remote.missing)} shard(s) upstream")
        transfer = Transfer(
            id=f"{remote.repo}/{remote.name}".replace("/", "_"),
            repo=remote.repo, name=remote.name, destination=str(destination),
            total_bytes=remote.size_bytes, files_total=len(remote.files),
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
        try:
            destination.mkdir(parents=True, exist_ok=True)
            for item in remote.files:
                if transfer._cancel.is_set():
                    transfer.state = State.CANCELLED
                    return
                self._file(transfer, remote.repo, item, destination)
                transfer.files_done += 1
            transfer.state = State.DONE
        except Exception as error:
            if transfer._cancel.is_set():
                transfer.state = State.CANCELLED
                transfer.error = ""
            else:
                transfer.state = State.FAILED
                transfer.error = str(error) or error.__class__.__name__
        # A finished transfer also means a new model on disk.
        self._announce(force=True)
        if self._bus is not None and transfer.state is State.DONE:
            self._bus.publish(ChangeEvent(topic="models"))

    def _file(self, transfer: Transfer, repo: str, item: RemoteFile,
              destination: Path) -> None:
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
