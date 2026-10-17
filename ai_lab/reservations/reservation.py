"""One reservation: who asked, how urgently, and where it stands.

Times are wall-clock seconds (`time.time()`), so a reservation saved before a
manager restart still means the same moment after it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields

from .settings import HIGHEST, LOWEST

QUEUED, GRANTED, ENDED = "queued", "granted", "ended"
TEXT_LIMIT = 200


@dataclass
class Reservation:
    """A request for the whole card, from asking to ending."""

    id: str
    team: str
    priority: int
    max_s: float
    requested_at: float
    last_beat: float
    instance: str = ""
    purpose: str = ""
    run_id: str = ""
    state: str = QUEUED
    granted_at: float | None = None
    first_call_at: float | None = None
    ended_at: float | None = None
    reason: str = ""
    restored: bool = False

    @property
    def active(self) -> bool:
        """Queued or holding the card."""
        return self.state != ENDED

    def ends_at(self) -> float | None:
        """When a granted reservation runs out of time."""
        return None if self.granted_at is None else self.granted_at + self.max_s

    def waited_levels(self, now: float, aging_s: int) -> int:
        """How many priority levels waiting has earned it so far."""
        if self.state != QUEUED or not aging_s:
            return 0
        return int((now - self.requested_at) // aging_s)

    def effective_priority(self, now: float, aging_s: int) -> int:
        """Its priority plus what waiting has earned, never above the highest."""
        return min(HIGHEST, self.priority + self.waited_levels(now, aging_s))

    def end(self, now: float, reason: str) -> None:
        """Stop it for good, saying why."""
        self.state, self.ended_at, self.reason = ENDED, now, reason

    def record(self) -> dict:
        """Everything, as saved to disk."""
        return asdict(self)

    @classmethod
    def from_record(cls, raw: dict) -> Reservation:
        """Back from disk; unknown keys are ignored."""
        known = {item.name for item in fields(cls)}
        return cls(**{key: value for key, value in raw.items() if key in known})


def checked_priority(value) -> int:
    """A priority from 1 (low) to 5 (high)."""
    if isinstance(value, bool) or not isinstance(value, int) or not LOWEST <= value <= HIGHEST:
        raise ValueError(f"priority must be a whole number from {LOWEST} to {HIGHEST}")
    return value


def checked_text(body: dict, key: str, required: bool = False) -> str:
    """A short label from the request (team, instance, purpose, run id)."""
    value = body.get(key, "")
    if not isinstance(value, str) or len(value) > TEXT_LIMIT:
        raise ValueError(f"{key} must be text of at most {TEXT_LIMIT} characters")
    if required and not value.strip():
        raise ValueError(f"{key} is required")
    return value.strip()
