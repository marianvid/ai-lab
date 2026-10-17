"""Who holds the card, who waits for it, and in what order (ADR 0005).

One reservation holds the whole card at a time. The others wait in a queue
ordered by priority, then by arrival; a waiting reservation rises one level
every `aging_s`, so nothing waits forever. A running reservation is never
interrupted: priority only decides who gets the card next.

A reservation ends when its holder releases it, when it stops sending
heartbeats, when it was granted and never used, when it reaches its maximum
duration, or when someone cancels it from the page. Every change is saved, so
a manager restart restores the book as it was (`restored: true`).

The front door calls `admit` before any call reaches a model: a call under the
holder's reservation goes ahead; a call under any other reservation is
refused with the reason; a call without one waits while the card is reserved.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path

from .errors import (
    CardReservedError,
    ReservationEndedError,
    ReservationNotGrantedError,
    ReservationRequiredError,
)
from .reservation import GRANTED, QUEUED, Reservation, checked_priority, checked_text
from .settings import ReservationPolicy
from .store import ReservationFile

HISTORY = 20
WAIT_STEP_S = 5.0
REAP_EVERY_S = 5.0

Clock = Callable[[], float]
StillWanted = Callable[[], bool]


class ReservationBook:
    """The card's reservations: the holder, the queue and the recent history."""

    def __init__(self, policy: ReservationPolicy, path: Path | None = None,
                 clock: Clock = time.time) -> None:
        self.policy = policy
        self._clock = clock
        self._file = ReservationFile(path)
        self._changed = threading.Condition()
        self._items: list[Reservation] = self._restored(self._file.load())

    # -- what a team does -----------------------------------------------------

    def request(self, body: dict) -> dict:
        """A new reservation: granted at once if the card is free, else queued."""
        now = self._clock()
        max_s = body.get("max_s", self.policy.default_max_s)
        reservation = Reservation(
            id=uuid.uuid4().hex[:12], team=checked_text(body, "team", required=True),
            instance=checked_text(body, "instance"), purpose=checked_text(body, "purpose"),
            run_id=checked_text(body, "run_id"),
            priority=checked_priority(body.get("priority", 3)),
            max_s=self._duration(max_s), requested_at=now, last_beat=now)
        with self._changed:
            self._items.append(reservation)
            self._settle_locked()
            return self._view(reservation)

    def get(self, reservation_id: str) -> dict:
        """Where one reservation stands now."""
        with self._changed:
            self._settle_locked()
            return self._view(self._find(reservation_id))

    def heartbeat(self, reservation_id: str) -> dict:
        """The holder (or a waiting team) is still alive."""
        with self._changed:
            self._settle_locked()
            reservation = self._live(reservation_id)
            reservation.last_beat = self._clock()
            return self._view(reservation)

    def extend(self, reservation_id: str, seconds) -> dict:
        """More time, up to the ceiling."""
        if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or seconds <= 0:
            raise ValueError("seconds must be a positive number")
        with self._changed:
            self._settle_locked()
            reservation = self._live(reservation_id)
            reservation.max_s = min(float(self.policy.ceiling_s), reservation.max_s + seconds)
            self._save_locked()
            return self._view(reservation)

    def release(self, reservation_id: str) -> dict:
        """The team is done: the card goes to the next in line."""
        return self._end(reservation_id, "released by its holder")

    def cancel(self, reservation_id: str) -> dict:
        """Stopped by hand from the AI-Lab page."""
        return self._end(reservation_id, "cancelled from the AI-Lab page")

    # -- what the front door asks ---------------------------------------------

    def admit(self, reservation_id: str | None, still_wanted: StillWanted | None = None) -> None:
        """Let a model call through, wait, or refuse it with the reason."""
        if reservation_id:
            self._admit_holder(reservation_id)
        elif self.policy.required:
            raise ReservationRequiredError(
                "this AI-Lab requires a card reservation: send X-AI-Lab-Reservation")
        else:
            self._wait_until_free(still_wanted)

    def holder(self) -> dict | None:
        """The reservation holding the card, if any."""
        with self._changed:
            self._settle_locked()
            held = self._granted()
            return None if held is None else self._view(held)

    # -- the page -------------------------------------------------------------

    def overview(self) -> dict:
        """Holder, queue in order, recent history and the settings."""
        with self._changed:
            self._settle_locked()
            held = self._granted()
            ended = [item for item in self._items if not item.active]
            return {"holder": None if held is None else self._view(held),
                    "queue": [self._view(item) for item in self._queue()],
                    "history": [self._view(item) for item in reversed(ended)],
                    "settings": self.policy.json()}

    def apply_policy(self, policy: ReservationPolicy) -> None:
        """New settings, used at once."""
        with self._changed:
            self.policy = policy
            self._settle_locked()

    def settle(self) -> None:
        """End what is due and hand the card on; called by the reaper too."""
        with self._changed:
            self._settle_locked()

    def start_reaper(self, every_s: float = REAP_EVERY_S) -> threading.Thread:
        """Check for ended reservations in the background, so a silent card is freed."""
        def loop() -> None:
            while True:
                time.sleep(every_s)
                self.settle()
        thread = threading.Thread(target=loop, daemon=True, name="reservations-reaper")
        thread.start()
        return thread

    # -- inside ---------------------------------------------------------------

    def _admit_holder(self, reservation_id: str) -> None:
        with self._changed:
            self._settle_locked()
            reservation = self._find_for_call(reservation_id)
            if reservation.state == QUEUED:
                position = self._queue().index(reservation) + 1
                raise ReservationNotGrantedError(
                    f"reservation {reservation_id} is not granted yet "
                    f"(position {position} in the queue)", {"reservation": self._view(reservation)})
            if reservation.first_call_at is None:
                reservation.first_call_at = self._clock()
                self._save_locked()

    def _find_for_call(self, reservation_id: str) -> Reservation:
        found = next((item for item in self._items if item.id == reservation_id), None)
        if found is None:
            raise ReservationEndedError(f"unknown reservation {reservation_id}")
        if not found.active:
            raise ReservationEndedError(f"reservation {reservation_id} has ended: {found.reason}",
                                   {"reservation": self._view(found)})
        return found

    def _wait_until_free(self, still_wanted: StillWanted | None) -> None:
        deadline = self._clock() + self.policy.unreserved_wait_s
        with self._changed:
            while True:
                self._settle_locked()
                held = self._granted()
                if held is None:
                    return
                if self._clock() >= deadline or (still_wanted and not still_wanted()):
                    raise CardReservedError(_reserved_message(held), {"holder": self._view(held)})
                self._changed.wait(min(WAIT_STEP_S, max(0.0, deadline - self._clock())))

    def _end(self, reservation_id: str, reason: str) -> dict:
        with self._changed:
            reservation = self._find(reservation_id)
            if reservation.active:
                reservation.end(self._clock(), reason)
            self._settle_locked(changed=True)
            return self._view(reservation)

    def _settle_locked(self, changed: bool = False) -> None:
        now = self._clock()
        for reservation in self._items:
            reason = self._due(reservation, now)
            if reason:
                reservation.end(now, reason)
                changed = True
        if self._granted() is None and self._queue():
            nxt = self._queue()[0]
            nxt.state, nxt.granted_at = GRANTED, now
            changed = True
        if changed:
            self._trim()
            self._save_locked()
            self._changed.notify_all()

    def _due(self, reservation: Reservation, now: float) -> str:
        policy = self.policy
        if not reservation.active:
            return ""
        if now - reservation.last_beat > policy.heartbeat_timeout_s:
            return f"no heartbeat for {policy.heartbeat_timeout_s} s"
        if reservation.state != GRANTED or reservation.granted_at is None:
            return ""
        if reservation.first_call_at is None and now - reservation.granted_at > policy.unused_s:
            return f"granted but unused for {policy.unused_s} s"
        if now >= reservation.granted_at + reservation.max_s:
            return "reached its maximum duration"
        return ""

    def _queue(self) -> list[Reservation]:
        now, aging = self._clock(), self.policy.aging_s
        waiting = [item for item in self._items if item.state == QUEUED]
        return sorted(waiting, key=lambda item: (-item.effective_priority(now, aging),
                                                 item.requested_at))

    def _granted(self) -> Reservation | None:
        return next((item for item in self._items if item.state == GRANTED), None)

    def _find(self, reservation_id: str) -> Reservation:
        found = next((item for item in self._items if item.id == reservation_id), None)
        if found is None:
            raise KeyError(f"Unknown reservation: {reservation_id}")
        return found

    def _live(self, reservation_id: str) -> Reservation:
        reservation = self._find(reservation_id)
        if not reservation.active:
            raise ReservationEndedError(f"reservation {reservation_id} has ended: {reservation.reason}",
                                   {"reservation": self._view(reservation)})
        return reservation

    def _duration(self, value) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)) \
                or not 0 < value <= self.policy.ceiling_s:
            raise ValueError(f"max_s must be a number of seconds from 1 to "
                             f"{self.policy.ceiling_s} (the ceiling)")
        return float(value)

    def _trim(self) -> None:
        ended = [item for item in self._items if not item.active]
        for old in ended[:-HISTORY]:
            self._items.remove(old)

    def _save_locked(self) -> None:
        self._file.save(self._items)

    def _restored(self, saved: list[Reservation]) -> list[Reservation]:
        now = self._clock()
        for reservation in saved:
            if reservation.active:
                reservation.restored = True
                reservation.last_beat = now   # the holder could not beat while we were down
        return saved

    def _view(self, reservation: Reservation) -> dict:
        now = self._clock()
        view = reservation.record()
        view["effective_priority"] = reservation.effective_priority(now, self.policy.aging_s)
        ends_at = reservation.ends_at()
        view["time_left_s"] = (None if ends_at is None or not reservation.active
                               else max(0, round(ends_at - now)))
        queue = self._queue()
        view["position"] = queue.index(reservation) + 1 if reservation in queue else None
        return view


def _reserved_message(held: Reservation) -> str:
    who = held.team + (f" @ {held.instance}" if held.instance else "")
    return f"the card is reserved by {who}" + (f" ({held.purpose})" if held.purpose else "")
