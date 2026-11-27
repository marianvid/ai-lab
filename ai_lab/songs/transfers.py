"""A song pulled from another AI-Lab, or put back from a kept bundle.

A pull fetches the bundle through this machine's scheduler door,
`{scheduler}/ai/<peer>/v1/audio/music/generations/{id}/bundle`, carrying the
caller's `X-Run-Id` (the run the door checks) and no token; the AI-Labs never
talk directly. The door's own refusals (its `scheduler` key) are told apart
from the other AI-Lab's answers. Neither ever deletes the source: the caller
(music) does that once the answer says `checked: true`.
"""

from __future__ import annotations

import json
import secrets
import shutil
import tarfile
import urllib.error
import urllib.request
from collections.abc import Iterator
from http import HTTPStatus
from pathlib import Path
from typing import Protocol
from urllib.parse import quote, urlencode

from ..shared.song_names import check_generation_id
from .bundles import BUNDLE_TYPE, into_place, manifest, receive, tar_pieces
from .errors import PeerConflictError, SchedulerDoorError, SongAlreadyHereError, TransferCheckError

# The run a pull belongs to: checked by the scheduler door, passed on unchanged.
RUN_HEADER = "X-Run-Id"


class Songs(Protocol):
    """The songs of a YuE2 entry, as `music_windows.GenerationFiles` keeps them."""

    def engine(self, query: dict | None):
        """The entry's engine, with `transfer` and `generations_dir()`."""
        ...

    def folder(self, query: dict | None, generation_id: str) -> Path:
        """The generation's folder; FileNotFoundError when there is none."""
        ...

    def refuse_busy(self, query: dict | None, generation_id: str) -> None:
        """409 while a window call or a Studio job works on the generation."""
        ...


def bundle_pieces(files: Songs, query: dict | None, generation_id: str) -> Iterator[bytes]:
    """The generation as a bundle, piece by piece; 409 while a window call or a Studio job works on it."""
    folder = files.folder(query, generation_id)
    files.refuse_busy(query, generation_id)
    listing = manifest(folder, files.engine(query).transfer.chunk_bytes)
    return tar_pieces(folder, listing)


class Pull:
    """One pull: the peer, the song, and the engine's folder it goes into.

    It only copies: the peer's song is left as it is. The caller (music) deletes the other
    copy itself once it has the answer.
    """

    def __init__(self, files: Songs, query: dict | None, generation_id: str, body,
                 run_id: str | None = None) -> None:
        if not run_id:
            raise ValueError(f"a pull needs the header {RUN_HEADER}: the run it belongs to, "
                             "which the scheduler door checks")
        self.run_id = run_id
        body = body if isinstance(body, dict) else {}
        unknown = set(body) - {"from"}
        if unknown:
            raise ValueError("Unknown pull fields: " + ", ".join(sorted(unknown)))
        self.engine = files.engine(query)
        self.transfer = self.engine.transfer
        self.peer = self.transfer.peer(body.get("from"))
        self.model = (query or {})["model"]
        self.generation_id = check_generation_id(generation_id)
        self.target = self.engine.generations_dir() / self.generation_id
        files.refuse_busy(query, self.generation_id)
        if self.target.exists():
            raise SongAlreadyHereError(f"{self.generation_id} is already on this machine")

    def run(self) -> dict:
        """Fetch the bundle, check every file, move the copy into place."""
        temporary = self.target.parent / f".pull-{self.generation_id}-{secrets.token_hex(4)}"
        try:
            with self._fetch() as response:
                count, size = receive(response, temporary, self.transfer.chunk_bytes,
                                      self.generation_id)
            into_place(temporary, self.target)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return {"generation_id": self.generation_id, "machine": self.transfer.machine,
                "files": count, "bytes": size, "checked": True}

    def _fetch(self):
        """The bundle, through this machine's scheduler door (no token: the door asks none)."""
        url = (f"{self.peer.url}/v1/audio/music/generations/{quote(self.generation_id, safe='')}"
               f"/bundle?{urlencode({'model': self.model})}")
        request = urllib.request.Request(url, method="GET", headers={RUN_HEADER: self.run_id})
        try:
            return urllib.request.urlopen(request, timeout=self.transfer.timeout_s)
        except urllib.error.HTTPError as error:
            raise self._refused(error.code, error.read().decode(errors="replace")) from None
        except (urllib.error.URLError, OSError) as error:
            raise SchedulerDoorError(f"the scheduler door at {self.transfer.scheduler} did not answer: "
                                     f"{getattr(error, 'reason', error)}") from None

    def _refused(self, code: int, text: str) -> Exception:
        """The error for a refused fetch: the door's own refusal, or the peer AI-Lab's answer."""
        door = _door_refusal(text)
        if door is not None:
            return SchedulerDoorError(f"the scheduler door refused the call to {self.peer.name} "
                                      f"(HTTP {code}, {door})")
        if code == HTTPStatus.NOT_FOUND:
            return FileNotFoundError(f"{self.peer.name} has no generation {self.generation_id}")
        if code == HTTPStatus.CONFLICT:
            return PeerConflictError(f"{self.peer.name} refused: {text[:500]}")
        return RuntimeError(f"{self.peer.name} answered HTTP {code}: {text[:500]}")


def _door_refusal(text: str) -> str | None:
    """`reason: detail` when an answer is the scheduler door's own refusal (its `scheduler` key)."""
    try:
        door = json.loads(text).get("scheduler")
    except (ValueError, AttributeError):
        return None
    if not isinstance(door, dict):
        return None
    return f"{door.get('reason', '?')}: {door.get('detail', '')}"


class PutBack:
    """One bundle put back into this machine's Studio, under the id it was taken from.

    The caller kept the bundle (`GET …/bundle`) when the song left the Studio; this checks
    it as a pull checks a fetched one. A bundle that does not match is refused (400) and
    nothing of it is kept.
    """

    def __init__(self, files: Songs, query: dict | None, generation_id: str, body,
                 run_id: str | None = None) -> None:
        """`body` is the bundle as it is read (with its `length`), or None when none was sent."""
        if not run_id:
            raise ValueError(f"a put-back needs the header {RUN_HEADER}: the run it belongs to")
        if body is None:
            raise ValueError(f"a put-back's body is the bundle, sent as {BUNDLE_TYPE}")
        self.engine = files.engine(query)
        self.transfer = self.engine.transfer
        most = self.transfer.put_back_most_bytes
        if not 0 < body.length <= most:
            raise ValueError(f"a bundle must be 1 to {most} bytes (put_back_most_bytes); "
                             f"this one says {body.length}")
        self.body = body
        self.generation_id = check_generation_id(generation_id)
        self.target = self.engine.generations_dir() / self.generation_id
        files.refuse_busy(query, self.generation_id)
        if self.target.exists():
            raise SongAlreadyHereError(f"{self.generation_id} is already on this machine")

    def run(self) -> dict:
        """Check every file against the manifest, then move the folder into place."""
        temporary = self.target.parent / f".put-back-{self.generation_id}-{secrets.token_hex(4)}"
        try:
            try:
                count, size = receive(self.body, temporary, self.transfer.chunk_bytes,
                                      self.generation_id)
            except (TransferCheckError, tarfile.TarError, ValueError) as error:
                raise ValueError(f"the bundle was refused: {error}") from None
            into_place(temporary, self.target)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return {"generation_id": self.generation_id, "machine": self.transfer.machine,
                "files": count, "bytes": size, "checked": True}
