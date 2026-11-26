"""The last known usage-limit state of each vendor, learned only from real calls.

What the console's band shows (`providers.<vendor>.limit` on `/api/gateway`):
`limited` while a vendor refuses calls for its allowance, `ok` again at the
next call that answers, `unknown` before any call. Like the logins, kept in a
file beside the usage counters, so a restart does not forget it.

- `kind` is `usage` (the subscription's allowance) or `rate` (a short
  refusal). A usage refusal is `limited` at once; a rate refusal only from
  the vendor's `limited_after_strikes`-th refusal in a row, so a single short
  pause does not light the band.
- `retry_at` is AI-Lab's own next try (the end of the vendor's pause);
  `resets_at` the vendor's own reset time, only when its words carry one
  AI-Lab can read (an ISO time, or Claude Code's `…|<epoch seconds>`).
"""

from __future__ import annotations

import calendar
import re
import threading
import time
from collections.abc import Callable
from pathlib import Path

from . import state_file

FILE_NAME = "limits.json"
# An ISO time, in UTC or with its offset, as a vendor may name its reset.
_ISO = re.compile(r"\b(\d{4}-\d{2}-\d{2})[T ](\d{2}:\d{2}(?::\d{2})?)(Z|[+-]\d{2}:?\d{2})?")
# Claude Code ends its usage-limit line with a bar and the reset in epoch seconds.
_EPOCH = re.compile(r"\|(\d{10})\b")


def utc(seconds: float) -> str:
    """A wall-clock time as `YYYY-MM-DDTHH:MM:SSZ`."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(seconds))


def resets_at(words: str) -> str:
    """The vendor's own reset time, when its words carry one AI-Lab can read; else empty."""
    epoch = _EPOCH.search(words or "")
    if epoch:
        return utc(int(epoch.group(1)))
    iso = _ISO.search(words or "")
    if not iso:
        return ""
    day, clock, zone = iso.groups()
    if clock.count(":") == 1:  # no seconds given
        clock += ":00"
    seconds = calendar.timegm(time.strptime(f"{day}T{clock}", "%Y-%m-%dT%H:%M:%S"))
    if zone not in (None, "Z"):
        sign = 1 if zone[0] == "+" else -1
        seconds -= sign * (int(zone[1:3]) * 3600 + int(zone[-2:]) * 60)
    return utc(seconds)


class LimitBook:
    """Per vendor: `ok` / `limited` / `unknown`, since when, and what the vendor said."""

    def __init__(self, state_dir: Path, clock: Callable[[], float] = time.time) -> None:
        self._path = Path(state_dir) / FILE_NAME
        self._clock = clock
        self._lock = threading.Lock()
        self._states: dict[str, dict] = self._read()

    def answered(self, vendor: str, model: str) -> None:
        """A call to this vendor answered: no limit now."""
        with self._lock:
            before = self._states.get(vendor) or {}
            if before.get("state") == "ok":
                return
            self._states[vendor] = {"state": "ok", "kind": "", "since": utc(self._clock()),
                                    "retry_at": "", "resets_at": "", "model": model, "said": ""}
            self._write()

    def limited(self, vendor: str, model: str, kind: str, said: str,
                retry_in_s: float) -> dict:
        """The vendor refused this call for a limit; returns the state now kept."""
        now = self._clock()
        with self._lock:
            before = self._states.get(vendor) or {}
            entry = {"state": "limited", "kind": kind,
                     "since": before["since"] if before.get("state") == "limited" else utc(now),
                     "retry_at": utc(now + retry_in_s) if retry_in_s > 0 else "",
                     "resets_at": resets_at(said), "model": model, "said": said}
            self._states[vendor] = entry
            self._write()
            return dict(entry)

    def state(self, vendor: str) -> dict:
        """What the page and the console show; `unknown` before any call."""
        with self._lock:
            return dict(self._states.get(vendor) or {"state": "unknown"})

    def _read(self) -> dict[str, dict]:
        return state_file.read(self._path)

    def _write(self) -> None:
        state_file.write(self._path, self._states)
