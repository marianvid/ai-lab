"""Everything about access in one object, built from the host's state folder.

The routes and the server talk to this, never to the parts directly, so that
the HTTP layer stays one call per request.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from .door import Door
from .presence import Presence
from .registry import ClientRegistry
from .security import SecuritySettings
from .sessions import SessionBook, cookie_header, session_of

CLIENTS_FILE = "clients.json"
SECURITY_FILE = "security.json"
ENDED = 0


class Access:
    """The clients, the door, the web sessions and Settings → Security."""

    def __init__(self, registry: ClientRegistry, security: SecuritySettings,
                 presence: Presence | None = None, sessions: SessionBook | None = None,
                 log: Callable[[str], None] = print) -> None:
        self.registry = registry
        self.security = security
        self.presence = presence or Presence()
        self.sessions = sessions or SessionBook()
        self.door = Door(registry, self.presence, security, self.sessions, log)

    @classmethod
    def in_folder(cls, state_dir: Path, log: Callable[[str], None] = print) -> Access:
        """The access kept in `state_dir` (clients.json, security.json)."""
        state_dir = Path(state_dir)
        return cls(ClientRegistry(state_dir / CLIENTS_FILE),
                   SecuritySettings(state_dir / SECURITY_FILE), log=log)

    # -- clients ------------------------------------------------------------

    def clients(self) -> dict:
        """Every registered client, with when it was last heard from and its map."""
        rows = self.registry.names()
        seen = {row["name"]: row for row in self.presence.view([r["name"] for r in rows])}
        return {"clients": [{**row, **seen[row["name"]]} for row in rows]}

    def register(self, name: str) -> dict:
        """A new client and its token, shown this once."""
        return {"name": name, "token": self.registry.register(name),
                "note": "Keep this token: AI-Lab shows it only now."}

    def remove(self, name: str) -> None:
        """Forget a client."""
        self.registry.remove(name)

    def heartbeat(self, headers, body: dict) -> float:
        """Keep the calling scheduler's map; the time received is returned."""
        name = self.door.caller(headers)
        if name is None:
            raise PermissionError("A heartbeat needs a registered client's token")
        return self.presence.heartbeat(name, body.get("map"))

    # -- the web interface --------------------------------------------------

    def session(self, headers) -> dict:
        """Whether the page must show the login form."""
        return {"login_required": self.security.login_required,
                "logged_in": self.sessions.valid(session_of(headers))}

    def login(self, password: str) -> str:
        """A Set-Cookie value for a right password; PermissionError otherwise."""
        if not self.security.check(password):
            raise PermissionError("Wrong password")
        return cookie_header(self.sessions.start())

    def logout(self, headers) -> str:
        """End this browser's session; the Set-Cookie value that clears it."""
        self.sessions.end(session_of(headers))
        return cookie_header("", ENDED)

    def page_cookie(self, headers) -> str | None:
        """With login off, the page opens a session by itself; else nothing."""
        if self.security.login_required or self.sessions.valid(session_of(headers)):
            return None
        return cookie_header(self.sessions.start())

    def update_security(self, changes: dict) -> dict:
        """Change the switches; switching login on ends every open session."""
        view = self.security.update(changes)
        if changes.get("login_required"):
            self.sessions.end_all()
        return view

    def set_password(self, new: str, current: str | None) -> dict:
        """Set or change the password; with login on, every open session ends."""
        self.security.set_password(new, current)
        if self.security.login_required:
            self.sessions.end_all()
        return self.security.view()
