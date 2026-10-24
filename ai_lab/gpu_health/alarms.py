"""Alarms: what is wrong, when to say so, and when to say it is over.

Two kinds of thing go wrong:
- a *condition*: true for a while, then false again (the chip too hot, the
  PCIe link slower than normal, R above its reference). It is announced when
  it starts, repeated while it lasts, and a "resolved" message is sent when it
  ends;
- an *event*: something that happened once (a new Xid in the kernel log, a
  new uncorrectable memory error, the power limit changed).

Anti-spam: a lasting alarm is repeated at most once every 6 hours
(`repeat_s`). An alarm that comes back after its "Resolved" message, or gets
worse (warning → alarm → critical), is sent at once.
Critical ones are never held back: every critical event is sent, and a
critical condition that persists is repeated every 15 minutes
(`critical_repeat_s`). Every alarm is written to the events table whether it
was sent or held back, so nothing is lost, only not repeated.

Levels: critical > alarm > warning > info.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field

from .settings import GpuHealthSettings
from .store import Store

LEVELS = ("info", "warning", "alarm", "critical")
ACTIVE_KEY = "alarms_active"
SENT_KEY = "alarms_last_sent"


@dataclass(frozen=True)
class Condition:
    """One alarm: a stable `key` (what it is about), a level, a sentence, details."""

    key: str
    level: str
    message: str
    detail: dict = field(default_factory=dict)


class AlarmBook:
    """Decides which alarms are sent, and keeps what is active across restarts."""

    def __init__(self, store: Store, settings: GpuHealthSettings,
                 send: Callable[[dict], None]) -> None:
        self._store = store
        self._settings = settings
        self._send = send
        self._active: dict[str, dict] = store.state(ACTIVE_KEY, {})
        self._sent: dict[str, float] = store.state(SENT_KEY, {})
        # The monitor's thread and the page (test alarm, status) both use the book.
        self._lock = threading.RLock()

    def active(self) -> list[dict]:
        """The conditions in force now, worst first."""
        with self._lock:
            return sorted((dict(item) for item in self._active.values()),
                          key=lambda item: (-LEVELS.index(item["level"]), item["since"]))

    def conditions(self, family: str, now: float, found: Iterable[Condition]) -> None:
        """The conditions of one family (`fast`, `slow`, `trend`, ...) that are true now.

        A family's condition missing from `found` has ended and is resolved.
        """
        by_key = {condition.key: condition for condition in found}
        with self._lock:
            for key, known in list(self._active.items()):
                if known["family"] == family and key not in by_key:
                    self._resolve(now, key)
            for condition in by_key.values():
                self._raise(family, now, condition)
            self._store.set_state(ACTIVE_KEY, self._active)

    def _resolve(self, now: float, key: str) -> None:
        """A condition ended. "Resolved" is sent when the alarm itself was: one
        held back by the anti-spam ends as quietly as it started."""
        known = self._active.pop(key)
        self._record(now, "resolved", Condition(key, known["level"],
                     "Resolved: " + known["message"], known.get("detail", {})),
                     known.get("notified", False))

    def _raise(self, family: str, now: float, condition: Condition) -> None:
        """A condition true now: new (sent unless held back), or lasting (repeated when due)."""
        entry = self._active.get(condition.key)
        due = self._due(condition.key, condition.level, now)
        # A condition that got worse is news: it never waits for the old level's timer.
        if entry is not None and LEVELS.index(condition.level) > LEVELS.index(entry["level"]):
            due = True
        if entry is None:
            due = due or condition.level == "critical"
            self._active[condition.key] = {**asdict(condition), "family": family,
                                           "since": now, "notified": due}
            self._record(now, "alarm", condition, due)
            return
        entry.update(level=condition.level, message=condition.message,
                     detail=condition.detail, notified=entry.get("notified", False) or due)
        self._record(now, "alarm", condition, due, repeat=True)

    def event(self, now: float, condition: Condition) -> None:
        """Something that happened once; critical ones always go out."""
        with self._lock:
            due = condition.level == "critical" or self._due(condition.key, condition.level, now)
            self._record(now, "event", condition, due)

    def _due(self, key: str, level: str, now: float) -> bool:
        """Whether an alarm may be sent now; "info" is shown, never sent."""
        if level == "info":
            return False
        wait = self._settings.critical_repeat_s if level == "critical" else self._settings.repeat_s
        return now - self._sent.get(key, float("-inf")) >= wait

    def _record(self, now: float, kind: str, condition: Condition, send: bool,
                repeat: bool = False) -> None:
        """Write the alarm to the events table; send it if it is due."""
        if repeat and not send:
            return
        event = {"ts": now, "kind": kind, "level": condition.level, "key": condition.key,
                 "message": condition.message,
                 "detail": {**condition.detail, "sent": send, "repeat": repeat}}
        self._store.add_event(event)
        if send:
            # After "Resolved" the clock starts over: if the condition comes
            # back, that is news and is sent, however soon it happens.
            if kind == "resolved":
                self._sent.pop(condition.key, None)
            else:
                self._sent[condition.key] = now
            self._store.set_state(SENT_KEY, self._sent)
            self._send(event)
