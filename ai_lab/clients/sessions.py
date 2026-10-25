"""Web sessions: the cookie that lets the interface in instead of a token.

In memory: a restart asks for the password again, which on a home network is
an acceptable price for keeping nothing about sessions on disk.
"""

from __future__ import annotations

import secrets
import threading
import time
from collections.abc import Callable
from http.cookies import CookieError, SimpleCookie

COOKIE = "ai_lab_session"
SESSION_BYTES = 32
LIFETIME_S = 30 * 24 * 3600.0


def cookie_header(session_id: str, max_age_s: float = LIFETIME_S) -> str:
    """The Set-Cookie value: not readable by scripts, not sent by other sites."""
    return (f"{COOKIE}={session_id}; Path=/; Max-Age={int(max_age_s)}; "
            "HttpOnly; SameSite=Strict")


def session_of(headers) -> str | None:
    """The session id in a request's Cookie header, if any."""
    raw = headers.get("Cookie") if headers is not None else None
    if not raw:
        return None
    try:
        jar = SimpleCookie(raw)
    except CookieError:
        return None
    morsel = jar.get(COOKIE)
    return morsel.value if morsel is not None else None


class SessionBook:
    """Session ids and when each expires."""

    def __init__(self, clock: Callable[[], float] = time.time,
                 lifetime_s: float = LIFETIME_S) -> None:
        self.clock = clock
        self.lifetime_s = lifetime_s
        self._lock = threading.Lock()
        self._expiry: dict[str, float] = {}

    def start(self) -> str:
        """A new session id."""
        session_id = secrets.token_urlsafe(SESSION_BYTES)
        with self._lock:
            self._expiry[session_id] = self.clock() + self.lifetime_s
        return session_id

    def valid(self, session_id: str | None) -> bool:
        """Whether this id names a session that has not expired."""
        if not session_id:
            return False
        with self._lock:
            expiry = self._expiry.get(session_id)
            if expiry is not None and expiry < self.clock():
                del self._expiry[session_id]
                return False
            return expiry is not None

    def end(self, session_id: str | None) -> None:
        """Forget a session (logout)."""
        with self._lock:
            self._expiry.pop(session_id or "", None)

    def end_all(self) -> None:
        """Forget every session (the password changed, or login was switched on)."""
        with self._lock:
            self._expiry.clear()
