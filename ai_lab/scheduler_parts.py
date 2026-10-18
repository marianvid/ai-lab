"""What the scheduler hands out and keeps: its errors, and one waiting request.

Kept apart from `scheduler.py` so the queue's rules (there) and the way a run
is chosen and published (`scheduler_planning.py`) can share them without
importing each other.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field

# Asked before a waiting request is let in: true while it is still wanted.
# It may also raise `AdmissionWithdrawnError` to refuse it with a reason.
StillWanted = Callable[[], bool]


class QueueFullError(RuntimeError):
    """More requests are waiting than this manager will hold."""


class WillNotFitError(RuntimeError):
    """This model does not fit on this machine, whatever else is unloaded.

    Told to the request rather than discovered by emptying the machine and
    failing anyway: unloading what was working, to load something that was
    never going to fit, is the worst of both.

    `detail` is filled in by whoever knows the numbers — the scheduler does
    not — and reaches the client as fields it can act on rather than a
    sentence it would have to parse.
    """

    kind = "insufficient_memory"
    code = "model_does_not_fit"

    def __init__(self, message: str, detail: dict | None = None) -> None:
        super().__init__(message)
        self.detail = detail or {}


class AbandonedError(RuntimeError):
    """The client went away before its turn came."""


class AdmissionWithdrawnError(AbandonedError):
    """The request may no longer be let in, for a reason of its own.

    Raised by a `StillWanted` callable at the head of the queue — for example
    when the card was reserved while the request waited (ADR 0005). The
    scheduler knows nothing of why: it drops the request before anything is
    unloaded and hands back `reason`, which the caller turns into its answer.
    An abandonment, not a fault of this machine.
    """

    def __init__(self, reason: BaseException) -> None:
        super().__init__(str(reason))
        self.reason = reason


@dataclass(slots=True)
class Waiting:
    """One request, from arriving to being let in."""

    number: int                       # arrival order, and the tie-break
    shape: object                     # opaque; equal shapes may run together
    arrived: float
    admitted: threading.Event = field(default_factory=threading.Event)
    # Set when this one will never be admitted, and why.
    failure: BaseException | None = None
    # Asked before it is let in. A client that has gone is dropped rather than
    # served, which matters most when serving it means a forty-second load.
    still_wanted: StillWanted | None = None

    def fail(self, failure: BaseException) -> None:
        """Wake it with the reason it will never be let in."""
        self.failure = failure
        self.admitted.set()


def release(ready: list[Waiting]) -> None:
    """Wake every request let in. Called after the scheduler's lock is dropped."""
    for entry in ready:
        entry.admitted.set()
