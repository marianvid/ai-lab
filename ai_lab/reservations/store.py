"""Reservations kept on disk, so a manager restart does not free a held card.

One JSON file, rewritten whole and atomically on every change. A file that
cannot be read is set aside rather than trusted, and the book starts empty.
"""

from __future__ import annotations

import json
from pathlib import Path

from .reservation import Reservation


class ReservationFile:
    """The book's state on disk: the live reservations and the recent history."""

    def __init__(self, path: Path | None) -> None:
        self.path = path

    def load(self) -> list[Reservation]:
        """What was saved, oldest first; empty when nothing was."""
        if self.path is None or not self.path.is_file():
            return []
        try:
            raw = json.loads(self.path.read_text())
            return [Reservation.from_record(item) for item in raw.get("reservations", [])]
        except (OSError, ValueError, TypeError):
            self.path.replace(self.path.with_suffix(".unreadable"))
            return []

    def save(self, reservations: list[Reservation]) -> None:
        """Write them all, replacing the file in one step."""
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(
            {"reservations": [item.record() for item in reservations]}, indent=1))
        temporary.replace(self.path)
