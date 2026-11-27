"""The bundle a song travels as, and checking one that arrived.

An uncompressed tar whose first member is `MANIFEST.json`: `{"generation_id",
"files": [{"name", "size", "sha256"}]}`; the files follow, flat, in the
manifest's order. A received bundle is written into a hidden folder, every file
checked against the manifest, and only then renamed into place.
"""

from __future__ import annotations

import hashlib
import json
import os
import tarfile
from collections.abc import Iterator
from io import BytesIO
from pathlib import Path
from typing import BinaryIO, cast

from ..shared.song_names import check_generation_id
from .errors import SongAlreadyHereError, TransferCheckError

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
    """Copy one member, refusing it as soon as it runs past the size its manifest declares."""
    digest, size, declared = hashlib.sha256(), 0, item["size"]
    with target.open("wb") as out:
        # One byte past the declared size is enough to know it is too large.
        while piece := source.read(min(chunk_bytes, declared + 1 - size)):
            size += len(piece)
            if size > declared:
                raise TransferCheckError(f"{item['name']} is larger than its manifest says "
                                         f"({declared} bytes)")
            digest.update(piece)
            out.write(piece)
    if size != item["size"] or digest.hexdigest() != item["sha256"]:
        raise TransferCheckError(f"{item['name']} arrived different from the source "
                                 f"({size} bytes, expected {item['size']})")
    return size


def into_place(temporary: Path, target: Path) -> None:
    """Rename the checked folder into place; 409 when another put-back or pull got there first."""
    if target.exists():
        raise SongAlreadyHereError(f"{target.name} appeared on this machine meanwhile")
    try:
        os.rename(temporary, target)
    except OSError as error:
        if target.exists():
            raise SongAlreadyHereError(f"{target.name} appeared on this machine meanwhile") from error
        raise


