"""The log of what AI-Lab is doing (A9): on screen, and on disk with rotation.

One line per thing worth knowing while it happens: a model loading or being
taken off for another, a call refused, AI-Lab switched Off. The page reads the
newest lines (`since`), the file keeps them, in a few files of bounded size.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from logging.handlers import RotatingFileHandler
from pathlib import Path

KEEP_ON_SCREEN = 500
FILE_BYTES = 5 * 1024 * 1024
FILE_COUNT = 5
LOG_NAME = "operations.log"
TIME_FORMAT = "%Y-%m-%d %H:%M:%S"


class OperationJournal:
    """The newest lines in memory, every line in a rotating file."""

    def __init__(self, folder: Path | None, clock: Callable[[], float] = time.time,
                 keep: int = KEEP_ON_SCREEN) -> None:
        self.clock = clock
        self._lock = threading.Lock()
        self._lines: deque[dict] = deque(maxlen=keep)
        self._next_id = 1
        self._file = _file_logger(folder) if folder is not None else None

    def write(self, text: str) -> None:
        """Add one line, now."""
        now = self.clock()
        with self._lock:
            self._lines.append({"id": self._next_id, "at": now, "text": text})
            self._next_id += 1
        if self._file is not None:
            stamp = time.strftime(TIME_FORMAT, time.gmtime(now))
            self._file.info("%s UTC %s", stamp, text)

    def since(self, after: int = 0) -> dict:
        """The lines newer than `after`, oldest first, and the newest id."""
        with self._lock:
            lines = [line for line in self._lines if line["id"] > after]
            return {"lines": lines, "last_id": self._next_id - 1}


class PhaseFollower:
    """Turns the runtime's progress events into one line per step of a load or stop.

    Progress events arrive many times a second while a model loads; only a
    change of step is worth a line.
    """

    def __init__(self, journal: OperationJournal) -> None:
        self.journal = journal
        self._last: dict[str, str] = {}

    def take(self, event) -> None:
        """Note the event if it starts a new step for its model."""
        phase = getattr(event, "phase", None)
        instance_id = getattr(event, "instance_id", None)
        if phase is None or instance_id is None:
            return
        name = getattr(phase, "value", str(phase))
        if self._last.get(instance_id) == name:
            return
        self._last[instance_id] = name
        message = getattr(event, "message", "")
        self.journal.write(f"{instance_id}: {name.replace('_', ' ')}"
                           + (f" — {message}" if message else ""))

    def follow(self, bus) -> threading.Thread:
        """Read `bus` in a thread of its own for as long as the process lives."""
        subscription = bus.subscribe()

        def run() -> None:
            for event in subscription.events():
                if event is not None:
                    self.take(event)

        thread = threading.Thread(target=run, daemon=True, name="operation-journal")
        thread.start()
        return thread


def _file_logger(folder: Path) -> logging.Logger:
    """A logger of its own that writes only to the rotating file."""
    Path(folder).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"ai_lab.journal.{folder}")
    logger.propagate = False
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = RotatingFileHandler(Path(folder) / LOG_NAME, maxBytes=FILE_BYTES,
                                      backupCount=FILE_COUNT, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    return logger
