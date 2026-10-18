"""How much room something takes on disk, how much is left, and what a file holds.

Used by every page that lists environments, builds and reclaimable storage.
Both answers are approximate on purpose: a folder may be deleted while it is
being counted (a half-built environment being removed at the same moment the
settings page asks how big everything is), and an approximate size for a
folder on its way out is the right answer where an exception that empties the
page is not.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path


def tree_size(path: Path) -> int:
    """Bytes in a file, or in every regular file under a folder; links are not followed."""
    if path.is_file() and not path.is_symlink():
        try:
            return path.stat().st_size
        except OSError:
            return 0
    total = 0
    try:
        for item in path.rglob("*"):
            try:
                if item.is_file() and not item.is_symlink():
                    total += item.stat().st_size
            except OSError:
                continue
    except OSError:
        return total
    return total


def free_bytes(path: Path | None) -> int:
    """Free space on the disk holding `path` (or its nearest existing parent); 0 if unknown."""
    if path is None:
        return 0
    place = path
    while not place.exists() and place != place.parent:
        place = place.parent
    try:
        return shutil.disk_usage(place).free
    except OSError:
        return 0


def sha256_of(path: Path) -> str:
    """The SHA-256 of a file's contents, read a chunk at a time, as hex."""
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()
