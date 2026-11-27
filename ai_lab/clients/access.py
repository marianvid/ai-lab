"""Everything about access in one object, built from the host's state folder.

The routes and the server talk to this, never to the parts directly, so that
the HTTP layer stays one call per request.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from ..config.policy import ClientCapLimits
from .caps import ClientCaps
from .door import Door, DoorKeys
from .errors import AccessDeniedError
from .mailbox import OWNER, Mailbox, check_client_name, party_of
from .power import Power
from .presence import Presence
from .registry import ClientRegistry
from .security import SecuritySettings
from .sessions import PAGE_SESSION_S, SessionBook, cookie_header, session_of

CLIENTS_FILE = "clients.json"
MAILBOX_FILE = "mailbox.jsonl"
CAPS_FILE = "client_caps.json"
SECURITY_FILE = "security.json"
POWER_FILE = "power.json"
SESSIONS_FILE = "sessions.json"
ENDED = 0


class Access:
    """The clients, the door, the web sessions, Settings → Security and On/Off."""

    def __init__(self, registry: ClientRegistry, security: SecuritySettings,
                 power: Power, log: Callable[[str], None] = print,
                 mailbox: Mailbox | None = None) -> None:
        self.registry = registry
        # Caps on a client's calls (`caps.py`); set by `in_folder`, absent in a test that needs none.
        self.caps: ClientCaps | None = None
        self.security = security
        self.power = power
        self.log = log
        self.mailbox = mailbox
        self.presence = Presence()
        # Kept beside security.json, so a login survives a restart ("remember me").
        self.sessions = SessionBook(lifetime_s=lambda: security.session_s,
                                    path=security.path.parent / SESSIONS_FILE)
        self.door = Door(DoorKeys(registry, self.presence, security, self.sessions), log, power)

    @classmethod
    def in_folder(cls, state_dir: Path, log: Callable[[str], None] = print,
                  caps: ClientCapLimits | None = None) -> Access:
        """The access kept in `state_dir` (clients.json, security.json, power.json, sessions.json).

        `caps`: how the client caps count and refuse (`limits.clients`).
        """
        state_dir = Path(state_dir)
        security = SecuritySettings(state_dir / SECURITY_FILE)
        access = cls(ClientRegistry(state_dir / CLIENTS_FILE), security,
                     Power(state_dir / POWER_FILE), log=log,
                     mailbox=Mailbox(state_dir / MAILBOX_FILE, security.mailbox_limits))
        access.caps = ClientCaps(state_dir / CAPS_FILE, caps)
        return access

    # -- clients ------------------------------------------------------------

    def clients(self) -> dict:
        """Every registered client, with its last call, its last heartbeat and its caps."""
        rows = self.registry.names()
        seen = {row["name"]: row for row in self.presence.view([r["name"] for r in rows])}
        caps = self.caps
        return {"clients": [{**row, **seen[row["name"]],
                             **(caps.view(row["name"]) if caps is not None else {})}
                            for row in rows]}

    def register(self, name: str) -> dict:
        """A new client and its token, shown this once."""
        check_client_name(name)
        return {"name": name, "token": self.registry.register(name),
                "note": "Keep this token: AI-Lab shows it only now."}

    def remove(self, name: str) -> None:
        """Forget a client, and its caps."""
        self.registry.remove(name)
        if self.caps is not None:
            self.caps.forget(name)

    def set_caps(self, name: str, changes: dict) -> dict:
        """The owner sets or clears a registered client's caps (`clients/caps.py`)."""
        if self.caps is None:
            raise RuntimeError("This AI-Lab was built without client caps")
        if name not in {row["name"] for row in self.registry.names()}:
            raise KeyError(f"Unknown client: {name}")
        return {"name": name, **self.caps.set(name, dict(changes))}

    def admit_call(self, headers, path: str, payload, default_cap: int):
        """Count a capped client's text call, or refuse it; the function that gives it back."""
        if self.caps is None:
            return lambda: None
        return self.caps.admit(self.door.caller(headers), path, payload, default_cap)

    def heartbeat(self, headers) -> float:
        """Note the calling client's heartbeat; the time received is returned.

        The body is not read: an older scheduler that still sends a `map` is
        answered the same, and the map is dropped.
        """
        name = self.door.caller(headers)
        if name is None:
            raise AccessDeniedError("A heartbeat needs a registered client's token")
        return self.presence.heartbeat(name)

    # -- the mailbox --------------------------------------------------------

    def mail_party(self, headers) -> str:
        """Who reads or writes the mailbox: a `mailbox-<party>` client, or the web interface as the owner.

        Asked with a token, never anonymously, whatever Settings → Security says:
        the mailbox is a conversation between named parties.
        """
        party = party_of(self.door.caller(headers))
        if party is not None:
            return party
        if self.sessions.valid(session_of(headers)):
            return OWNER
        raise AccessDeniedError("The mailbox needs a mailbox client's token "
                                "(a client named mailbox-<party>) or the web interface")

    def _box(self) -> Mailbox:
        if self.mailbox is None:
            raise RuntimeError("This AI-Lab was built without a mailbox")
        return self.mailbox

    def mail_read(self, headers, after: int) -> dict:
        """The messages after `after` that the caller may read."""
        return self._box().read(self.mail_party(headers), after)

    def mail_post(self, headers, to: str, text: str) -> dict:
        """One message from the caller; nothing in it is acted on."""
        return self._box().post(self.mail_party(headers), to, text)

    def switch(self, on: bool) -> dict:
        """Switch AI-Lab On or Off (A6), and say so in the log."""
        was = self.power.on
        view = self.power.switch(on)
        if was != view["on"]:
            self.log(f"AI-Lab switched {'On' if view['on'] else 'Off'}")
        return view

    # -- the web interface --------------------------------------------------

    def session(self, headers) -> dict:
        """Whether the page must show the login form."""
        return {"login_required": self.security.login_required,
                "logged_in": self.sessions.valid(session_of(headers)),
                "token_holds": self.security.token_holds}

    def login(self, password: str) -> str:
        """A Set-Cookie value for a right password; AccessDeniedError otherwise."""
        if not self.security.check(password):
            raise AccessDeniedError("Wrong password")
        return cookie_header(self.sessions.start(), self.sessions.lifetime_s)

    def logout(self, headers) -> str:
        """End this browser's session; the Set-Cookie value that clears it."""
        self.sessions.end(session_of(headers))
        return cookie_header("", ENDED)

    def page_cookie(self, headers) -> str | None:
        """With login off, the page opens a session by itself; else nothing."""
        if self.security.login_required or self.sessions.valid(session_of(headers)):
            return None
        return cookie_header(self.sessions.start(kept=False), PAGE_SESSION_S)

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
