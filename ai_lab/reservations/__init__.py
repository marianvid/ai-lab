"""Card reservations: a team holds the card for a whole run (ADR 0005).

`ReservationBook` keeps who holds the card, who waits and in what order, and
ends reservations whose holder stopped beating or ran out of time. The front
door asks it, through `admit`, before any call reaches a model.
"""

from .book import ReservationBook
from .errors import (
    CardReservedError,
    ReservationConflictError,
    ReservationEndedError,
    ReservationNotGrantedError,
    ReservationRequiredError,
)
from .settings import ReservationPolicy

__all__ = ["CardReservedError", "ReservationBook", "ReservationConflictError", "ReservationEndedError",
           "ReservationNotGrantedError", "ReservationPolicy", "ReservationRequiredError"]
