"""The registered clients and their tokens.

A token is shown once, when its client is registered, and never again: the
file keeps only its SHA-256 digest. A digest without a salt is enough here,
because a token is 32 random bytes, not a word a person chose.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
import threading
import time
from pathlib import Path

TOKEN_BYTES = 32
FILE_MODE = 0o600
NAME = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")


def digest(token: str) -> str:
    """The stored form of a token."""
    return hashlib.sha256(token.encode()).hexdigest()


def write_private(path: Path, payload: dict) -> None:
    """Write JSON readable by the owner only, in one atomic step."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, FILE_MODE)
    with os.fdopen(descriptor, "w") as stream:
        json.dump(payload, stream, indent=1, sort_keys=True)
    os.replace(temporary, path)


class ClientRegistry:
    """Clients by name, each with the digest of its token, kept in one private file."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self._clients: dict[str, dict] = self._read()

    def _read(self) -> dict[str, dict]:
        if not self.path.is_file():
            return {}
        data = json.loads(self.path.read_text() or "{}")
        return dict(data.get("clients", {}))

    def register(self, name: str) -> str:
        """A new token for `name`, replacing any earlier one. Returned only here."""
        if not NAME.match(name or ""):
            raise ValueError("A client name is lower-case letters, digits, '.', '_' or '-'")
        token = secrets.token_urlsafe(TOKEN_BYTES)
        with self._lock:
            self._clients[name] = {"digest": digest(token), "registered_at": time.time()}
            write_private(self.path, {"clients": self._clients})
        return token

    def remove(self, name: str) -> None:
        """Forget a client; its token stops working at once."""
        with self._lock:
            if name not in self._clients:
                raise KeyError(f"Unknown client: {name}")
            del self._clients[name]
            write_private(self.path, {"clients": self._clients})

    def names(self) -> list[dict]:
        """Every client, with when it was registered (never its token)."""
        with self._lock:
            return [{"name": name, "registered_at": row["registered_at"]}
                    for name, row in sorted(self._clients.items())]

    def identify(self, token: str) -> str | None:
        """The client this token belongs to, or None."""
        wanted = digest(token)
        with self._lock:
            for name, row in self._clients.items():
                if hmac.compare_digest(row["digest"], wanted):
                    return name
        return None
