"""Moving a YuE2 song (a generation) from one AI-Lab to another.

The machine that should have the song pulls it, so the other deletes it only after
the copy is known to be whole:

- `GET /v1/audio/music/generations/{id}/bundle?model=<m>` answers the song's folder as an
  uncompressed tar. Its first member is `MANIFEST.json`: `{"generation_id", "files":
  [{"name", "size", "sha256"}]}`; the files follow, flat, in the manifest's order.
- `POST /v1/audio/music/generations/{id}/pull?model=<m>` with `{"from": "<peer>"}` fetches
  that bundle from the peer named in `engines.yue2.transfer` (directly, AI-Lab to AI-Lab),
  writes it into a hidden folder beside the songs, checks every size and sha256 against the
  manifest, renames it into place, then deletes the song on the peer.

Both read and write the engine's folder, so they work whether YuE2 is loaded or not.
"""
from __future__ import annotations

import hashlib
import json
import os
import secrets
import shutil
import tarfile
import urllib.error
import urllib.request
from collections.abc import Iterator
from http import HTTPStatus
from io import BytesIO
from pathlib import Path
from typing import BinaryIO, Protocol, cast
from urllib.parse import quote, urlencode

from ...gateway.errors import TransferCheckError, WindowCallActiveError
from ...music.backends.yue2_web_backend import check_generation_id
from ..passthrough import Passthrough

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


MANIFEST = "MANIFEST.json"
# A manifest larger than this is not one (a song has about a dozen files).
LARGEST_MANIFEST_BYTES = 1 << 20
BUNDLE_TYPE = "application/x-tar"


def manifest(folder: Path, chunk_bytes: int) -> dict:
    """The generation's files with their sizes and sha256; refused if the folder is not flat."""
    files = []
    for path in sorted(folder.iterdir()):
        if path.is_symlink() or not path.is_file() or path.name.startswith("."):
            raise ValueError(f"{folder.name} holds {path.name}, which is not a plain file")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while piece := handle.read(chunk_bytes):
                digest.update(piece)
        files.append({"name": path.name, "size": path.stat().st_size, "sha256": digest.hexdigest()})
    return {"generation_id": folder.name, "files": files}


class _Sink:
    """Where `tarfile` writes a stream; what it wrote is taken out piece by piece."""

    def __init__(self) -> None:
        self.pieces: list[bytes] = []

    def write(self, data: bytes) -> int:
        self.pieces.append(bytes(data))
        return len(data)

    def take(self) -> Iterator[bytes]:
        pieces, self.pieces = self.pieces, []
        yield from pieces


def tar_pieces(folder: Path, listing: dict) -> Iterator[bytes]:
    """The bundle: the manifest first, then each file it lists."""
    sink = _Sink()
    with tarfile.open(fileobj=cast(BinaryIO, sink), mode="w|", format=tarfile.PAX_FORMAT) as tar:
        data = json.dumps(listing, indent=1).encode()
        info = tarfile.TarInfo(MANIFEST)
        info.size = len(data)
        tar.addfile(info, BytesIO(data))
        yield from sink.take()
        for item in listing["files"]:
            tar.add(folder / item["name"], arcname=item["name"], recursive=False)
            yield from sink.take()
    yield from sink.take()


def bundle(files: Songs, query: dict | None, generation_id: str) -> Passthrough:
    """The generation as a bundle; 409 while a window call or a Studio job works on it."""
    folder = files.folder(query, generation_id)
    files.refuse_busy(query, generation_id)
    listing = manifest(folder, files.engine(query).transfer.chunk_bytes)
    return Passthrough(status=HTTPStatus.OK, content_type=BUNDLE_TYPE, chunks=tar_pieces(folder, listing),
                       headers={"Content-Disposition": f'attachment; filename="{generation_id}.tar"'})


def receive(stream, temporary: Path, chunk_bytes: int) -> tuple[int, int]:
    """Unpack a bundle into `temporary`, checking each file; (files, bytes) when all match."""
    temporary.mkdir()
    with tarfile.open(fileobj=stream, mode="r|") as tar:
        first = tar.next()
        if first is None or first.name != MANIFEST or not first.isfile() \
                or first.size > LARGEST_MANIFEST_BYTES:
            raise TransferCheckError("the bundle does not start with its manifest")
        expected = _expected(json.load(_member(tar, first)))
        got: dict[str, int] = {}
        # `tar.next()`, not `for member in tar`: iterating a stream starts again
        # with the members already read (the manifest).
        while (member := tar.next()) is not None:
            item = expected.get(member.name)
            if item is None or not member.isfile() or member.name in got:
                raise TransferCheckError(f"the bundle holds {member.name!r}, which its manifest does not list")
            got[member.name] = _write_checked(_member(tar, member), temporary / member.name, item, chunk_bytes)
    missing = sorted(set(expected) - set(got))
    if missing:
        raise TransferCheckError(f"the bundle ended without: {', '.join(missing)}")
    return len(got), sum(got.values())


def _expected(listing) -> dict[str, dict]:
    """The manifest's files by name; refused when a name could leave the folder."""
    try:
        items = {check_generation_id(item["name"]): item for item in listing["files"]}
    except (KeyError, TypeError, ValueError):
        raise TransferCheckError("the bundle's manifest is not one") from None
    return items


def _member(tar: tarfile.TarFile, member: tarfile.TarInfo):
    handle = tar.extractfile(member)
    if handle is None:
        raise TransferCheckError(f"{member.name} cannot be read from the bundle")
    return handle


def _write_checked(source, target: Path, item: dict, chunk_bytes: int) -> int:
    digest, size = hashlib.sha256(), 0
    with target.open("wb") as out:
        while piece := source.read(chunk_bytes):
            digest.update(piece)
            size += len(piece)
            out.write(piece)
    if size != item["size"] or digest.hexdigest() != item["sha256"]:
        raise TransferCheckError(f"{item['name']} arrived different from the source "
                                 f"({size} bytes, expected {item['size']})")
    return size


def complete_song(folder: Path) -> bool:
    """Whether a song here looks whole: its audio is there and the Studio marked it complete.

    The cheap check before deleting the peer's copy on a retry; the full check (every
    size and sha256) was made when the song was pulled.
    """
    try:
        result = json.loads((folder / "result.json").read_text())
    except (OSError, ValueError):
        return False
    return (folder / "audio.flac").is_file() and isinstance(result, dict) \
        and result.get("status") == "complete"


class Pull:
    """One pull: the peer, the song, and the engine's folder it goes into.

    With `"retry_delete": true` it fetches nothing: the song is already here (pulled before,
    its peer's delete failed) and only the delete on the peer is sent again.
    """

    def __init__(self, files: Songs, query: dict | None, generation_id: str, body) -> None:
        body = body if isinstance(body, dict) else {}
        unknown = set(body) - {"from", "retry_delete"}
        if unknown:
            raise ValueError("Unknown pull fields: " + ", ".join(sorted(unknown)))
        self.retry_delete = body.get("retry_delete", False)
        if not isinstance(self.retry_delete, bool):
            raise ValueError("retry_delete must be true or false")
        self.engine = files.engine(query)
        self.transfer = self.engine.transfer
        self.peer = self.transfer.peer(body.get("from"))
        self.model = (query or {})["model"]
        self.generation_id = check_generation_id(generation_id)
        self.target = self.engine.generations_dir() / self.generation_id
        files.refuse_busy(query, self.generation_id)
        if self.retry_delete:
            self._check_kept_copy()
        elif self.target.exists():
            raise WindowCallActiveError(f"{self.generation_id} is already on this machine")

    def _check_kept_copy(self) -> None:
        if not self.target.is_dir():
            raise FileNotFoundError(f"{self.generation_id} is not on this machine: pull it first")
        if not complete_song(self.target):
            raise WindowCallActiveError(f"{self.generation_id} here is not a complete song; "
                                        f"the copy on {self.peer.name} is kept")

    def run(self) -> dict:
        """Fetch, check, move into place, then delete on the peer (only the delete on a retry)."""
        if self.retry_delete:
            return self._removed({"generation_id": self.generation_id, "machine": self.transfer.machine})
        temporary = self.target.parent / f".pull-{self.generation_id}-{secrets.token_hex(4)}"
        try:
            with self._ask("GET", "/bundle") as response:
                count, size = receive(response, temporary, self.transfer.chunk_bytes)
            if self.target.exists():
                raise WindowCallActiveError(f"{self.generation_id} appeared on this machine meanwhile")
            os.rename(temporary, self.target)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return self._removed({"generation_id": self.generation_id, "machine": self.transfer.machine,
                              "files": count, "bytes": size})

    def _removed(self, answer: dict) -> dict:
        """`answer` with whether the peer's copy is gone, and why not when it is not."""
        error = self._delete_on_peer()
        answer["source_removed"] = error is None
        if error is not None:
            answer["source_error"] = error
        return answer

    def _ask(self, method: str, suffix: str):
        url = (f"{self.peer.url}/v1/audio/music/generations/{quote(self.generation_id, safe='')}"
               f"{suffix}?{urlencode({'model': self.model})}")
        request = urllib.request.Request(url, method=method,
                                         headers={"Authorization": "Bearer " + self.peer.token()})
        try:
            return urllib.request.urlopen(request, timeout=self.transfer.timeout_s)
        except urllib.error.HTTPError as error:
            text = error.read().decode(errors="replace")[:500]
            if error.code == HTTPStatus.NOT_FOUND:
                raise FileNotFoundError(f"{self.peer.name} has no generation {self.generation_id}") from None
            if error.code == HTTPStatus.CONFLICT:
                raise WindowCallActiveError(f"{self.peer.name} refused: {text}") from None
            raise RuntimeError(f"{self.peer.name} answered HTTP {error.code}: {text}") from None

    def _delete_on_peer(self) -> str | None:
        """None when the peer's copy is gone (deleted now, or already: its 404), else why not."""
        try:
            with self._ask("DELETE", ""):
                return None
        except FileNotFoundError:
            return None
        except Exception as error:
            return str(error)


def register_transfers(router, files: Songs) -> None:
    """The bundle and pull routes."""
    base = "/v1/audio/music/generations/{generation_id}"
    router.add("GET", base + "/bundle",
               lambda generation_id, query=None, **_: bundle(files, query, generation_id))
    router.add("POST", base + "/pull",
               lambda generation_id, query=None, body=None, **_: Pull(files, query, generation_id, body).run())
