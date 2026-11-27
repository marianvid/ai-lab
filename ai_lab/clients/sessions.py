"""Web sessions: the cookie that lets the interface in instead of a token.

"Remember me" (the owner's decision on the read's m1): a session holds for its
lifetime (Settings → Security, `session_days`, 30 by default) through both
things that used to end it:
- the browser closing: the cookie has a `Max-Age`, so the browser keeps it;
- an AI-Lab restart (every deploy is one): the book is kept on disk in the
  private state folder. Only a SHA-256 digest of each session id is written,
  never the id itself, so the file cannot be used to log in.

Only a login's session is written. With login off, the page hands a session to
every `GET /` without a cookie (a browser, curl, a health check, a scanner).
Such a session protects nothing, so it is kept in memory only, for a day, and
at most PAGE_SESSIONS_MAX of them: a restart costs nothing but a new cookie.

Ending a session (log out), switching login on, or changing the password with
login on removes the digests from the file too.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import secrets
import threading
import time
from collections.abc import Callable
from http.cookies import CookieError, SimpleCookie
from pathlib import Path

from .registry import write_private

COOKIE = "ai_lab_session"
SESSION_BYTES = 32
DAY_S = 24 * 3600.0
LIFETIME_S = 30 * DAY_S
# A session the page opened by itself, with login off: short, in memory, capped.
PAGE_SESSION_S = DAY_S
PAGE_SESSIONS_MAX = 1000


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


def digest(session_id: str) -> str:
    """What the book keeps of a session id: its SHA-256, never the id."""
    return hashlib.sha256(session_id.encode()).hexdigest()


def _kept(path: Path | None) -> dict[str, float]:
    """The digests and expiries in `path`; nothing if it is missing or unreadable."""
    if path is None or not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text() or "{}")
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {key: float(value) for key, value in data.items()
            if isinstance(key, str) and isinstance(value, (int, float))}


class SessionBook:
    """Session digests and when each expires, kept in `path` when one is given."""

    def __init__(self, clock: Callable[[], float] = time.time,
                 lifetime_s: float | Callable[[], float] = LIFETIME_S,
                 path: Path | None = None) -> None:
        self.clock = clock
        self._lifetime = lifetime_s if callable(lifetime_s) else (lambda: float(lifetime_s))
        self.path = Path(path) if path is not None else None
        self._lock = threading.Lock()
        now = clock()
        self._expiry = {key: expiry for key, expiry in _kept(self.path).items()
                        if expiry >= now}
        self._page: dict[str, float] = {}

    @property
    def lifetime_s(self) -> float:
        """How long a new session holds, in seconds (Settings → Security)."""
        return self._lifetime()

    def start(self, kept: bool = True) -> str:
        """A new session id; only its digest is kept, on disk for a login (`kept`).

        A page session (`kept=False`) lasts PAGE_SESSION_S, in memory only;
        past PAGE_SESSIONS_MAX the oldest goes.
        """
        session_id = secrets.token_urlsafe(SESSION_BYTES)
        with self._lock:
            now = self.clock()
            if kept:
                self._expiry = {key: expiry for key, expiry in self._expiry.items()
                                if expiry >= now}
                self._expiry[digest(session_id)] = now + self.lifetime_s
                self._save()
                return session_id
            self._page = {key: expiry for key, expiry in self._page.items() if expiry >= now}
            while len(self._page) >= PAGE_SESSIONS_MAX:
                del self._page[min(self._page, key=self._page.__getitem__)]
            self._page[digest(session_id)] = now + PAGE_SESSION_S
        return session_id

    def valid(self, session_id: str | None) -> bool:
        """Whether this id names a session that has not expired."""
        if not session_id:
            return False
        key = digest(session_id)
        with self._lock:
            if key in self._page:
                if self._page[key] >= self.clock():
                    return True
                del self._page[key]
                return False
            expiry = self._expiry.get(key)
            if expiry is not None and expiry < self.clock():
                del self._expiry[key]
                self._save()
                return False
            return expiry is not None

    def end(self, session_id: str | None) -> None:
        """Forget a session (logout)."""
        with self._lock:
            self._page.pop(digest(session_id or ""), None)
            if self._expiry.pop(digest(session_id or ""), None) is not None:
                self._save()

    def end_all(self) -> None:
        """Forget every session (the password changed, or login was switched on)."""
        with self._lock:
            self._expiry.clear()
            self._page.clear()
            self._save()

    def __len__(self) -> int:
        with self._lock:
            return len(self._expiry) + len(self._page)

    def _save(self) -> None:
        """Best effort: a disk that refuses costs the sessions at the next restart, nothing now."""
        if self.path is None:
            return
        with contextlib.suppress(OSError):
            write_private(self.path, self._expiry)
