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

from .registry import write_private

SALT_BYTES = 16
SCRYPT = {"n": 2 ** 14, "r": 8, "p": 1}
MIN_PASSWORD = 8


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

    def view(self) -> dict:
        """What the Security page shows (never the hash)."""
        return {"require_token": self.require_token, "login_required": self.login_required,
                "password_set": self.password_set}

    def update(self, changes: dict) -> dict:
        """Change the switches; login cannot be required without a password."""
        unknown = set(changes) - {"require_token", "login_required"}
        if unknown:
            raise ValueError(f"Unknown security setting: {', '.join(sorted(unknown))}")
        if changes.get("login_required") and not self.password_set:
            raise ValueError("Set a password before requiring login")
        with self._lock:
            self._data.update({key: bool(value) for key, value in changes.items()})
            write_private(self.path, self._data)
        return self.view()

    def set_password(self, new: str, current: str | None = None) -> None:
        """Set the password; changing an existing one needs the current one."""
        if self.password_set and not self.check(current or ""):
            raise PermissionError("The current password is wrong")
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
