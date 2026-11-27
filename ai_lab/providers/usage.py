"""Calls per vendor per day, kept on disk, so the shared allowance is visible.

One file, `provider-usage.json` in the state directory::

    {"2026-10-07": {"anthropic": {"requests": 12, "ok": 11,
                                  "models": {"sonnet": 12},
                                  "failures": {"rate limit": 1}}}}

Requests, not tokens: the CLIs do not report tokens in a form used here. Only
the most recent days are kept.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from datetime import date
from pathlib import Path

from . import state_file

FILE_NAME = "provider-usage.json"
DAYS_KEPT = 31


class UsageLog:
    """Counts calls and failures; thread-safe; survives a restart."""

    def __init__(self, state_dir: Path,
                 today: Callable[[], date] = date.today) -> None:
        self.path = Path(state_dir) / FILE_NAME
        self._today = today
        self._lock = threading.Lock()
        self._days = self._read()

    def record(self, vendor: str, model: str, failure: str = "") -> None:
        """One call: counted as a success, or as a failure under its label."""
        with self._lock:
            day = self._days.setdefault(self._today().isoformat(), {})
            counts = day.setdefault(vendor, {"requests": 0, "ok": 0,
                                             "models": {}, "failures": {}})
            counts["requests"] += 1
            counts["models"][model] = counts["models"].get(model, 0) + 1
            if failure:
                counts["failures"][failure] = counts["failures"].get(failure, 0) + 1
            else:
                counts["ok"] += 1
            for old in sorted(self._days)[:-DAYS_KEPT]:
                del self._days[old]
            self._write()

    def today(self) -> dict:
        """Today's counts, per vendor."""
        with self._lock:
            return json.loads(json.dumps(self._days.get(self._today().isoformat(), {})))

    def _read(self) -> dict:
        try:
            loaded = json.loads(self.path.read_text())
        except (OSError, ValueError):
            return {}
        return loaded if isinstance(loaded, dict) else {}

    def _write(self) -> None:
        # Best effort, like the logins and limits (audit L39): a full or
        # read-only disk must not turn an answer already paid for into an error.
        state_file.write(self.path, self._days)
