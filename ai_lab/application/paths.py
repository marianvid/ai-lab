"""Choosing a path from the browser: listing a directory, and making one.

A web page cannot open a file dialog on the machine the server runs on, so
the server offers the listing itself. Folders always; files only when
programs are asked for, and then only ones that can be launched. Never file
contents: this says what is there, and the less it can reach the better.
"""

from __future__ import annotations

import os
from pathlib import Path

# A shared library carries the execute bit and cannot be launched. Measured in
# llama.cpp's build directory on the container: 125 executable files, 33 of
# them `.so` companions to the launchers beside them.
LIBRARIES = (".so", ".dylib", ".dll")


def browse(start: Path, programs: bool = False) -> dict:
    """What is inside one directory: its folders, and its programs if asked for."""
    start = start.expanduser().resolve()
    if not start.is_dir():
        raise ValueError(f"{start} is not a directory")
    try:
        entries = [entry for item in sorted(start.iterdir())
                   if not item.name.startswith(".")
                   and (entry := _entry(item, programs)) is not None]
    except PermissionError:
        raise ValueError(f"No permission to read {start}") from None
    return {
        "path": str(start),
        "parent": str(start.parent) if start.parent != start else None,
        "writable": os.access(start, os.W_OK | os.X_OK),
        "entries": entries,
    }


def _entry(item: Path, programs: bool) -> dict | None:
    """One row of the listing, or None for what is not shown."""
    if item.is_dir():
        return {"name": item.name, "path": str(item), "kind": "folder",
                "writable": os.access(item, os.W_OK | os.X_OK)}
    if (programs and item.is_file() and os.access(item, os.X_OK)
            and not _is_library(item.name)):
        return {"name": item.name, "path": str(item), "kind": "program",
                "writable": False}
    return None


def browse_root(repository_paths: list[str]) -> Path:
    """Start where the models already are, not at the root of the disk."""
    for text in repository_paths:
        path = Path(text)
        if path.is_dir():
            return path.parent
    return Path.home()


def create_directory(path: str) -> dict:
    """Make a directory, so a repository can point at somewhere new.

    Offered because the alternative is telling someone to go and use a
    terminal in the middle of filling in a form.
    """
    target = Path(str(path)).expanduser()
    if target.exists():
        if target.is_dir():
            return {"path": str(target.resolve()), "created": False}
        raise ValueError(f"{target} exists and is not a directory")
    try:
        target.mkdir(parents=True)
    except OSError as error:
        raise ValueError(f"Could not create {target}: {error}") from None
    return {"path": str(target.resolve()), "created": True}


def _is_library(name: str) -> bool:
    lowered = name.lower()
    return any(lowered.endswith(end) or f"{end}." in lowered for end in LIBRARIES)
