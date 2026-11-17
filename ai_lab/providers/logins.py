"""The last known state of each vendor's login, learned only from real calls.

AI-Lab never checks a login by itself (the owner, 2026-11-17: no daily
check). A call that answers says `ok`; a call the vendor refused for its
sign-in says `expired`; every other failure (a rate limit, a timeout, an
unknown model) says nothing about the login and changes nothing. Before the
first call after a start the state is what the file kept, or `unknown`.

Kept in `logins.json` beside the usage counters, so a restart does not forget
an expired login.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from pathlib import Path

FILE_NAME = "logins.json"


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class LoginBook:
    """Per vendor: `ok` / `expired` / `unknown`, since when, and the call that said so."""

    def __init__(self, state_dir: Path, clock: Callable[[], str] = _now) -> None:
        self._path = Path(state_dir) / FILE_NAME
        self._clock = clock
        self._lock = threading.Lock()
        self._states: dict[str, dict] = self._read()

    def answered(self, vendor: str, model: str) -> None:
        """A call to this vendor answered: its login works."""
        self._learn(vendor, model, "ok", "")

    def refused(self, vendor: str, model: str, said: str) -> None:
        """The vendor refused this call for its sign-in."""
        self._learn(vendor, model, "expired", said)

    def state(self, vendor: str) -> dict:
        """What the page and the console show; `unknown` before any call."""
        with self._lock:
            return dict(self._states.get(vendor) or {"state": "unknown"})

    def _learn(self, vendor: str, model: str, state: str, said: str) -> None:
        now = self._clock()
        with self._lock:
            before = self._states.get(vendor) or {}
            entry = {"state": state,
                     "since": before.get("since", now) if before.get("state") == state else now,
                     "last_learned": now, "model": model}
            if said:
                entry["said"] = said
            changed = before.get("state") != state or before.get("said") != entry.get("said")
            self._states[vendor] = entry
            if changed:
                self._write()

    def _read(self) -> dict[str, dict]:
        try:
            data = json.loads(self._path.read_text())
        except (OSError, ValueError):
            return {}
        return {key: value for key, value in data.items() if isinstance(value, dict)} \
            if isinstance(data, dict) else {}

    def _write(self) -> None:
        """Only when the state changes: an answered call does not write the file."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            spare = self._path.with_suffix(".tmp")
            spare.write_text(json.dumps(self._states, indent=1, sort_keys=True))
            spare.replace(self._path)
        except OSError:
            pass
