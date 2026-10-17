"""Why a call or a reservation was refused, each with what the caller needs to act."""

from __future__ import annotations


class ReservationConflictError(Exception):
    """The card's reservations stand in the way of this call (answered as 409)."""

    kind = "reservation_error"

    def __init__(self, message: str, detail: dict | None = None) -> None:
        super().__init__(message)
        self.detail = detail or {}


class ReservationEndedError(ReservationConflictError):
    """The call names a reservation that is over, or unknown."""


class ReservationNotGrantedError(ReservationConflictError):
    """The call names a reservation still waiting in the queue."""


class CardReservedError(ReservationConflictError):
    """Another team holds the card and the call could wait no longer."""


class ReservationRequiredError(Exception):
    """Reservations are mandatory and the call carries none (answered as 428)."""

    kind = "reservation_required"
