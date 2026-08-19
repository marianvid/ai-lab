"""Building model directories out of empty files.

A model is defined by its file names and sizes, so a whole catalogue can be
created in milliseconds without a single real weight.
"""

from __future__ import annotations

from pathlib import Path

from ai_lab.config import Repository


def make_files(directory: Path, *names: str, size: int = 16) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        (directory / name).write_bytes(b"x" * size)


def repository(root: Path, format: str = "gguf", identifier: str = "repo") -> Repository:
    return Repository(id=identifier, name=identifier, path=str(root), format=format)
