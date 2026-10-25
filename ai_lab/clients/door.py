"""The door every call passes: a token, a web session, or one of the open routes.

Two modes, switched in Settings → Security (`require_token`):
- off (the switch-over, step 1 of the approved order): a call without a valid
  token is still served, and the door logs it once per caller and route;
- on (step 4): such a call is refused with 401 and its code.

A token that belongs to nobody is the same in both modes as a missing one: it
is logged, or refused. The web interface is let in by its session cookie.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .errors import AiLabOffError, TokenInvalidError, TokenMissingError
from .power import Power, uses_a_model
from .presence import Presence
from .registry import ClientRegistry
from .security import SecuritySettings
from .sessions import SessionBook, session_of

# Reachable with nothing at all: "is it up", and what the login page needs.
OPEN = frozenset({("GET", "/api/health"), ("GET", "/api/session"),
                  ("POST", "/api/login")})
WEB = "web"
ANONYMOUS = "anonymous"
BEARER = "bearer "


def bearer(headers) -> str | None:
    """The token of an `Authorization: Bearer <token>` header, if any."""
    raw = headers.get("Authorization") if headers is not None else None
    if raw and raw.lower().startswith(BEARER):
        return raw[len(BEARER):].strip() or None
    return None


def route_key(path: str) -> str:
    """The first two segments of a path: enough to log a route once, not every id."""
    return "/" + "/".join(path.strip("/").split("/")[:2])


@dataclass(frozen=True)
class DoorKeys:
    """What the door checks a call against: tokens, sessions, and the switches."""

    registry: ClientRegistry
    presence: Presence
    security: SecuritySettings
    sessions: SessionBook


class Door:
    """Says who is calling, or refuses the call."""

    def __init__(self, keys: DoorKeys, log: Callable[[str], None] = print,
                 power: Power | None = None) -> None:
        self.registry = keys.registry
        self.presence = keys.presence
        self.security = keys.security
        self.sessions = keys.sessions
        self.log = log
        self.power = power
        self._logged: set[tuple[str, str]] = set()

    def admit(self, method: str, path: str, headers) -> str:
        """The caller's name, `web`, or `anonymous`; raises when refusing."""
        if (method.upper(), path) in OPEN:
            return ANONYMOUS
        caller = self._who(path, headers)
        if self.power is not None and not self.power.on and uses_a_model(method, path):
            raise AiLabOffError("AI-Lab is switched Off: it serves no model until it is "
                                "switched On", since=self.power.view()["since"])
        return caller

    def _who(self, path: str, headers) -> str:
        token = bearer(headers)
        if token is not None:
            name = self.registry.identify(token)
            if name is not None:
                self.presence.seen(name)
                return name
            return self._without(TokenInvalidError, "a token that is not registered", path)
        if self.sessions.valid(session_of(headers)):
            return WEB
        return self._without(TokenMissingError, "no token", path)

    def caller(self, headers) -> str | None:
        """The registered client behind a request's token, if any."""
        token = bearer(headers)
        return self.registry.identify(token) if token else None

    def _without(self, error: type, what: str, path: str) -> str:
        if self.security.require_token:
            raise error(f"AI-Lab refused the call: {what}. "
                        "Send 'Authorization: Bearer <token>'.")
        key = (what, route_key(path))
        if key not in self._logged:
            self._logged.add(key)
            self.log(f"[clients] call with {what} on {key[1]} (served: tokens not required yet)")
        return ANONYMOUS
