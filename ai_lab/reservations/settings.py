"""The `reservations` section of the configuration, checked.

Every number is a duration in seconds. All can be changed from the
Reservations page; the defaults are the ones the ADR settled on (ADR 0005).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

# (default, lowest, highest) per setting.
LIMITS = {
    "heartbeat_timeout_s": (180, 30, 3600),
    "unused_s": (600, 60, 86400),
    "default_max_s": (1800, 60, 86400),
    "ceiling_s": (14400, 60, 604800),
    "aging_s": (600, 0, 86400),
    "unreserved_wait_s": (1800, 0, 86400),
}
LOWEST, HIGHEST = 1, 5


@dataclass(frozen=True)
class ReservationPolicy:
    """How long a holder may go quiet, wait or hold, and whether reserving is required.

    - `heartbeat_timeout_s`: no heartbeat for this long ends a reservation.
    - `unused_s`: a granted reservation that makes no call for this long ends;
      after its first call only the heartbeat counts.
    - `default_max_s` / `ceiling_s`: how long one may hold the card when it does
      not say, and the most it may ever ask for, extensions included.
    - `aging_s`: a waiting reservation rises one priority level this often (0: never).
    - `unreserved_wait_s`: how long a call without a reservation waits while the
      card is reserved before it is refused.
    - `required`: refuse every card call that carries no reservation.
    """

    heartbeat_timeout_s: int = 180
    unused_s: int = 600
    default_max_s: int = 1800
    ceiling_s: int = 14400
    aging_s: int = 600
    unreserved_wait_s: int = 1800
    required: bool = False

    @classmethod
    def from_mapping(cls, raw: dict | None) -> ReservationPolicy:
        """The section as stored; absent keys take their defaults."""
        raw = raw or {}
        if not isinstance(raw, dict):
            raise ValueError("reservations must be an object")
        unknown = set(raw) - set(LIMITS) - {"required"}
        if unknown:
            raise ValueError(f"Unknown reservation settings: {', '.join(sorted(unknown))}")
        numbers = {key: _seconds(key, raw.get(key, default), low, high)
                   for key, (default, low, high) in LIMITS.items()}
        required = raw.get("required", False)
        if not isinstance(required, bool):
            raise ValueError("reservations.required must be true or false")
        if numbers["default_max_s"] > numbers["ceiling_s"]:
            raise ValueError("reservations.default_max_s may not exceed ceiling_s")
        return cls(required=required, **numbers)

    def json(self) -> dict:
        """The settings as the page and the configuration hold them."""
        return asdict(self)


def _seconds(key: str, value, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)) \
            or not low <= value <= high:
        raise ValueError(f"reservations.{key} must be a number of seconds "
                         f"from {low} to {high}")
    return int(value)
