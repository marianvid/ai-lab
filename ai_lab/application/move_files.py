"""The pure parts of a model move: its plan and the file-system checks.

Kept apart from `model_storage.py`, which owns the job records and the copy.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import ModelRoot, Repository


@dataclass(frozen=True)
class MovePlan:
    """What a move will do, settled before the job record is written."""

    model: Any
    model_id: str
    source_repository: Repository
    target_repository: Repository
    target_root: ModelRoot
    source_root: Path
    target_path: Path
    sources: list[Path]
    relatives: list[Path]
    destinations: list[Path]


def reject_unfit_destination(plan: MovePlan) -> None:
    """Refuse to overwrite anything, or to start a copy that cannot fit."""
    existing = [path for path in plan.destinations if path.exists()]
    if existing:
        raise ValueError(f"Destination already contains {existing[0]}")
    free = shutil.disk_usage(plan.target_path).free
    if free < plan.model.size_bytes:
        raise ValueError(
            f"{plan.target_root.name} does not have enough free space: "
            f"needs {plan.model.size_bytes} bytes, has {free}")


def reject_unremovable_sources(sources: list[Path]) -> None:
    """Refuse before copying when the manager cannot remove the source.

    A manually installed model may be readable while its directory is
    owned by root.  Discovering that only after copying and checksumming
    tens of gigabytes leaves two complete copies and a failed move.
    """
    blocked = sorted({path.parent for path in sources
                      if not os.access(path.parent, os.W_OK)})
    if blocked:
        raise ValueError(
            "The model is readable but cannot be moved because the "
            f"manager cannot remove files from {blocked[0]}. Correct its "
            "ownership or permissions, then try again.")


def prune_empty(paths: list[Path], roots: list[Path]) -> None:
    """Take away the directory too, if the model was the only thing in it.

    A model usually lives in its own directory, and leaving empty ones
    behind makes the library look like it still holds something.
    """
    for directory in {path.parent for path in paths}:
        if directory in roots:
            continue
        if any(directory.iterdir()):
            continue
        if any(directory.is_relative_to(root) for root in roots):
            directory.rmdir()
