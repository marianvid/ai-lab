"""Moving a YuE2 song (a generation) from one AI-Lab to another.

The machine that should have the song pulls it; the copy is kept only once it is
known to be whole:

- `GET /v1/audio/music/generations/{id}/bundle?model=<m>` answers the song's folder as an
  uncompressed tar. Its first member is `MANIFEST.json`: `{"generation_id", "files":
  [{"name", "size", "sha256"}]}`; the files follow, flat, in the manifest's order.
- `POST /v1/audio/music/generations/{id}/pull?model=<m>` with `{"from": "<peer>"}` and the
  header `X-Run-Id` fetches that bundle from the peer named in `engines.yue2.transfer`,
  writes it into a hidden folder beside the songs, checks every size and sha256 against the
  manifest, and renames it into place. It never deletes the peer's copy: the caller
  (music) does that itself once the answer says `checked: true`.
- `GET /v1/audio/music/generations?model=<m>[&limit=<n>][&offset=<k>]` lists this machine's
  generations, newest first, with what recognises each one. A read: nothing is changed.
- `POST /v1/audio/music/generations/{id}/put-back?model=<m>` with a bundle as the body
  (`Content-Type: application/x-tar`, the same tar `bundle` answers) and the header
  `X-Run-Id` puts a kept song back: every file is checked against the manifest as a pull
  checks it, written into a hidden folder, and renamed into place under the same id.
  Refused, never overwritten, when that id is already here or busy.

The AI-Labs never talk directly: the fetch goes through this machine's scheduler door,
`{scheduler}/ai/<peer>/v1/audio/music/generations/{id}/bundle`, carrying the caller's
`X-Run-Id` (the run the door checks) and no token. A pull without `X-Run-Id` is refused
(400). The door's own refusals (its `scheduler` key) are told apart from the other AI-Lab's
answers: a door's 404 is never taken for "the peer has no such song".

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
from datetime import datetime, timezone
from http import HTTPStatus
from io import BytesIO
from pathlib import Path
from typing import BinaryIO, Protocol, cast
from urllib.parse import quote, urlencode

from ...gateway.errors import SchedulerDoorError, TransferCheckError, WindowCallActiveError
from ...music.backends.yue2_web_backend import check_generation_id
from ..passthrough import Passthrough
from ..raw_body import RawBody


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
# The run a pull belongs to: checked by the scheduler door, passed on unchanged.
RUN_HEADER = "X-Run-Id"


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


def receive(stream, temporary: Path, chunk_bytes: int,
            generation_id: str | None = None) -> tuple[int, int]:
    """Unpack a bundle into `temporary`, checking each file; (files, bytes) when all match.

    With `generation_id`, the manifest must name that generation.
    """
    temporary.mkdir()
    with tarfile.open(fileobj=stream, mode="r|") as tar:
        expected = _read_manifest(tar, generation_id)
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


def _read_manifest(tar: tarfile.TarFile, generation_id: str | None) -> dict[str, dict]:
    """The bundle's first member, its manifest, as the files it lists by name."""
    first = tar.next()
    if first is None or first.name != MANIFEST or not first.isfile() \
            or first.size > LARGEST_MANIFEST_BYTES:
        raise TransferCheckError("the bundle does not start with its manifest")
    listing = json.load(_member(tar, first))
    expected = _expected(listing)
    if generation_id is not None and listing.get("generation_id") != generation_id:
        raise TransferCheckError(f"the bundle is {listing.get('generation_id')!r}, "
                                 f"not {generation_id!r}")
    return expected


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
            raise WindowCallActiveError(f"{self.generation_id} is already on this machine")

    def run(self) -> dict:
        """Fetch the bundle, check every file, move the copy into place."""
        temporary = self.target.parent / f".pull-{self.generation_id}-{secrets.token_hex(4)}"
        try:
            with self._fetch() as response:
                count, size = receive(response, temporary, self.transfer.chunk_bytes,
                                      self.generation_id)
            if self.target.exists():
                raise WindowCallActiveError(f"{self.generation_id} appeared on this machine meanwhile")
            os.rename(temporary, self.target)
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
            return WindowCallActiveError(f"{self.peer.name} refused: {text[:500]}")
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


# What a list entry quotes from the song's request, at most (characters).
QUOTED_CHARS = 300


def generations(files: Songs, query: dict | None) -> dict:
    """This machine's generations, newest first, `?limit=` (at most `list_most`) from `?offset=`."""
    engine = files.engine(query)
    most = engine.transfer.list_most
    limit = _count(query, "limit", most, 1, most)
    offset = _count(query, "offset", 0, 0, None)
    root = engine.generations_dir()
    folders = [path for path in root.iterdir()
               if path.is_dir() and not path.name.startswith(".")] if root.is_dir() else []
    # Sorted by time first, so only the page's folders are read in detail. A folder that
    # vanishes meanwhile (deleted while this runs) is left out rather than failing the list.
    dated = sorted(((when, path) for path in folders if (when := _made(path)) is not None),
                   key=lambda item: (item[0], item[1].name), reverse=True)
    page = [entry for _when, path in dated[offset:offset + limit] if (entry := _entry(path)) is not None]
    return {"machine": engine.transfer.machine, "total": len(dated), "offset": offset,
            "limit": limit, "generations": page}


def _made(folder: Path) -> float | None:
    """When the generation was finished (its `result.json`, else the folder); None if it is gone."""
    for path in (folder / "result.json", folder):
        try:
            return path.stat().st_mtime
        except FileNotFoundError:
            continue
    return None


def _count(query: dict | None, key: str, default: int, lowest: int, highest: int | None) -> int:
    raw = (query or {}).get(key)
    if raw is None:
        return default
    if not raw.isdigit() or int(raw) < lowest or (highest is not None and int(raw) > highest):
        top = "" if highest is None else f" to {highest}"
        raise ValueError(f"{key} must be a whole number from {lowest}{top}")
    return int(raw)


def _entry(folder: Path) -> dict | None:
    """One generation: its id, when it was made, its size, and what recognises it; None if it is gone."""
    try:
        made = _made(folder)
        sizes = [path.stat().st_size for path in folder.iterdir()
                 if path.is_file() and not path.name.startswith(".")]
    except FileNotFoundError:
        return None
    if made is None:
        return None
    request = _json_file(folder / "request.json")
    result = _json_file(folder / "result.json")
    sources = result.get("sources") if isinstance(result.get("sources"), dict) else {}
    seconds = result.get("audio_seconds")
    return {
        "generation_id": folder.name,
        "made_at": _iso(made),
        "bytes": sum(sizes),
        "files": len(sizes),
        "complete": result.get("status") == "complete",
        "kind": "window" if sources else "song",
        "source": sources.get("song") if sources else None,
        "seconds": round(float(seconds), 2) if isinstance(seconds, (int, float)) else None,
        "style": _quote(request.get("style")),
        "lyrics_start": _quote(request.get("lyrics")),
        "seed": request.get("seed"),
        "cot": request.get("cot"),
        "cfg_scale": request.get("cfg_scale"),
    }


def _json_file(path: Path) -> dict:
    """A small JSON file of the Studio's as a dictionary; empty when missing or unreadable."""
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _quote(value) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value if len(value) <= QUOTED_CHARS else value[:QUOTED_CHARS] + "…"


def _iso(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat(timespec="seconds")


class PutBack:
    """One bundle put back into this machine's Studio, under the id it was taken from.

    The caller kept the bundle (`GET …/bundle`) when the song left the Studio; this checks
    it as a pull checks a fetched one. A bundle that does not match is refused (400) and
    nothing of it is kept.
    """

    def __init__(self, files: Songs, query: dict | None, generation_id: str, body,
                 run_id: str | None = None) -> None:
        if not run_id:
            raise ValueError(f"a put-back needs the header {RUN_HEADER}: the run it belongs to")
        if not isinstance(body, RawBody):
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
            raise WindowCallActiveError(f"{self.generation_id} is already on this machine")

    def run(self) -> dict:
        """Check every file against the manifest, then move the folder into place."""
        temporary = self.target.parent / f".put-back-{self.generation_id}-{secrets.token_hex(4)}"
        try:
            try:
                count, size = receive(self.body, temporary, self.transfer.chunk_bytes,
                                      self.generation_id)
            except (TransferCheckError, tarfile.TarError, ValueError) as error:
                raise ValueError(f"the bundle was refused: {error}") from None
            _into_place(temporary, self.target)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return {"generation_id": self.generation_id, "machine": self.transfer.machine,
                "files": count, "bytes": size, "checked": True}


def _into_place(temporary: Path, target: Path) -> None:
    """Rename the checked folder into place; 409 when another put-back or pull got there first."""
    if target.exists():
        raise WindowCallActiveError(f"{target.name} appeared on this machine meanwhile")
    try:
        os.rename(temporary, target)
    except OSError as error:
        if target.exists():
            raise WindowCallActiveError(f"{target.name} appeared on this machine meanwhile") from error
        raise


def run_id_of(headers) -> str | None:
    """The caller's `X-Run-Id`, if it sent one."""
    value = headers.get(RUN_HEADER) if headers is not None else None
    if not isinstance(value, str):
        return None
    return value.strip() or None


def register_transfers(router, files: Songs) -> None:
    """The bundle and pull routes."""
    base = "/v1/audio/music/generations/{generation_id}"
    router.add("GET", base + "/bundle",
               lambda generation_id, query=None, **_: bundle(files, query, generation_id))
    router.add("POST", base + "/pull",
               lambda generation_id, query=None, body=None, headers=None, **_:
               Pull(files, query, generation_id, body, run_id_of(headers)).run())
    router.add("POST", base + "/put-back",
               lambda generation_id, query=None, body=None, headers=None, **_:
               PutBack(files, query, generation_id, body, run_id_of(headers)).run())
    router.add("GET", "/v1/audio/music/generations",
               lambda query=None, **_: generations(files, query))
