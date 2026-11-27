"""Settings → Security: the password, the "no logon" switch, the token switch.

The password is kept as a salted scrypt hash, never as text. Login can be
required only once a password exists, so switching it on can never lock
the owner out of an AI-Lab that has none.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import threading
from pathlib import Path

from .errors import AccessDeniedError
from ..shared.numbers import whole_number
from .registry import write_private

SALT_BYTES = 16
SCRYPT = {"n": 2 ** 14, "r": 8, "p": 1}
MIN_PASSWORD = 8
SWITCHES = ("require_token", "login_required")
# The agents' mailbox (clients/mailbox.py). A mailbox is a conversation, not an
# archive: 2 000 messages is weeks of two agents talking. 20 000 characters is a
# long report, and the size of the largest MS Graph result sqwad.ai quotes is
# what an agent may need to pass on in parts.
MAILBOX_LIMITS = {"mailbox_keep": (2000, 10, 100000), "mailbox_max_chars": (20000, 100, 200000)}
# How long a web session holds, in days ("remember me", `sessions.py`): the
# browser keeps the cookie that long and AI-Lab keeps it across restarts.
SESSION_LIMITS = {"session_days": (30, 1, 365)}
NUMBERS = {**MAILBOX_LIMITS, **SESSION_LIMITS}


def _hash(password: str, salt: bytes) -> str:
    return hashlib.scrypt(password.encode(), salt=salt, dklen=32, **SCRYPT).hex()


class SecuritySettings:
    """`require_token`, `login_required` and the password hash, in one private file."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self._data = json.loads(self.path.read_text() or "{}") if self.path.is_file() else {}

    @property
    def require_token(self) -> bool:
        """Whether a call without a valid token is refused (else only logged)."""
        return bool(self._data.get("require_token", False))

    @property
    def login_required(self) -> bool:
        """Whether the web interface asks for the password ("no logon" when False)."""
        return bool(self._data.get("login_required", False)) and self.password_set

    @property
    def password_set(self) -> bool:
        """Whether a password exists."""
        return "password" in self._data

    @property
    def token_holds(self) -> bool:
        """False only when a token is required with login off: then it protects nothing (m1)."""
        return not (self.require_token and not self.login_required)

    @property
    def session_s(self) -> float:
        """How long a new web session holds, in seconds."""
        return self.number("session_days") * 24 * 3600.0

    def number(self, key: str) -> int:
        """A number of `NUMBERS` (the mailbox's two, `session_days`): the value set, else its default."""
        return int(self._data.get(key, NUMBERS[key][0]))

    def mailbox_limits(self) -> tuple[int, int]:
        """(messages kept, characters a message may have), as the mailbox reads them at each post."""
        return self.number("mailbox_keep"), self.number("mailbox_max_chars")

    def view(self) -> dict:
        """What the Security page shows (never the hash)."""
        return {"require_token": self.require_token, "login_required": self.login_required,
                "password_set": self.password_set, "token_holds": self.token_holds,
                **{key: self.number(key) for key in NUMBERS}}

    def update(self, changes: dict) -> dict:
        """Change the switches and the mailbox's limits; login cannot be required without a password."""
        unknown = set(changes) - set(SWITCHES) - set(NUMBERS)
        if unknown:
            raise ValueError(f"Unknown security setting: {', '.join(sorted(unknown))}")
        if changes.get("login_required") and not self.password_set:
            raise ValueError("Set a password before requiring login")
        values: dict[str, bool | int] = {key: bool(changes[key]) for key in SWITCHES if key in changes}
        _refuse_half_protection(self.token_holds, {"require_token": self.require_token,
                                                   "login_required": self.login_required, **values})
        for key, (_default, low, high) in NUMBERS.items():
            if key in changes:
                number = whole_number(changes[key], key)
                if not low <= number <= high:
                    raise ValueError(f"{key} must be between {low} and {high}")
                values[key] = number
        with self._lock:
            self._data.update(values)
            write_private(self.path, self._data)
        return self.view()

    def set_password(self, new: str, current: str | None = None) -> None:
        """Set the password; changing an existing one needs the current one."""
        if self.password_set and not self.check(current or ""):
            raise AccessDeniedError("The current password is wrong")
        if len(new or "") < MIN_PASSWORD:
            raise ValueError(f"A password has at least {MIN_PASSWORD} characters")
        salt = secrets.token_bytes(SALT_BYTES)
        with self._lock:
            self._data["password"] = {"salt": salt.hex(), "hash": _hash(new, salt)}
            write_private(self.path, self._data)

    def check(self, password: str) -> bool:
        """Whether `password` is the one set."""
        stored = self._data.get("password")
        if not stored:
            return False
        typed = _hash(password or "", bytes.fromhex(stored["salt"]))
        return hmac.compare_digest(typed, stored["hash"])


HALF_PROTECTION_REFUSED = ("The token requirement only holds with login on: without it, the page "
                     "hands any visitor a session that is admitted as the web interface. "
                     "Set a password and turn login on first, or turn the token requirement off.")


def _refuse_half_protection(held_before: bool, after: dict) -> None:
    """Refuse a change that puts the token required with login off (the read's m1).

    In that combination `GET /` answers with a session cookie, and every later
    call carrying it is admitted as `web`: the token switch looks on and
    protects nothing. A change that turns the token off, or turns login on,
    is always taken. A machine already in that combination (both were, when
    this was written) is not changed and not locked out: its other settings
    still save, and every page warns until login is on.
    """
    if held_before and after.get("require_token") and not after.get("login_required"):
        raise ValueError(HALF_PROTECTION_REFUSED)
