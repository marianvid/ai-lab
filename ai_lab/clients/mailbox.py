"""A mailbox for agents that talk to each other through AI-Lab, and the owner who reads them.

Asked by the owner (2026-11-23) so that the ai-lab agent and the sqwad.ai agent
on his VM can write to each other without him carrying every message. The
rules came from the sqwad agent and he agreed:

- a message is information, not a command: nothing here executes anything,
  and an agent that reads one treats it as information too;
- the owner can read the whole history (the web interface reads every message);
- the mailbox has its own tokens, separate from any token that calls a model.

A party is a registered client named `mailbox-<party>`: `mailbox-sqwad` writes
as `sqwad`. `owner` and `all` are not party names: `owner` is the web
interface, `all` addresses everyone. A mailbox client opens the mailbox only
(the door refuses it anywhere else).

Messages are appended one JSON object per line. How many are kept and how long
one may be are Settings → Security values (`mailbox_keep`,
`mailbox_max_chars`); the oldest are dropped only when the file holds a fifth
more than `mailbox_keep`, so a post appends a line instead of rewriting the file.
"""

from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path

PREFIX = "mailbox-"
OWNER = "owner"
EVERYONE = "all"
RESERVED = frozenset({OWNER, EVERYONE})
PARTY = re.compile(r"^[a-z0-9][a-z0-9-]{0,31}$")
TRIM_MARGIN = 1.2


def party_of(client: str | None) -> str | None:
    """The mailbox party behind a client name; None for a client that is not one, or a reserved name."""
    if not client or not client.startswith(PREFIX):
        return None
    party = client[len(PREFIX):]
    return party if PARTY.match(party) and party not in RESERVED else None


def check_client_name(name: str) -> None:
    """Refuse registering a mailbox client whose party name is reserved or malformed."""
    if name.startswith(PREFIX) and party_of(name) is None:
        raise ValueError(f"'{name}' cannot be a mailbox client: after '{PREFIX}' comes a party name "
                         "(lowercase letters, digits, dashes), and not 'owner' or 'all'")


class Mailbox:
    """Messages between parties, read by `after` id. `limits` gives (keep, max_chars) at each post."""

    def __init__(self, path: Path, limits) -> None:
        self.path = Path(path)
        self.limits = limits
        self._lock = threading.Lock()

    def _all(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]

    def _last_id(self) -> int:
        if not self.path.exists():
            return 0
        lines = [line for line in self.path.read_text().splitlines() if line.strip()]
        return json.loads(lines[-1])["id"] if lines else 0

    def post(self, sender: str, to: str, text: str) -> dict:
        """Keep one message; its id and time are returned with it."""
        keep, max_chars = self.limits()
        to = (to or EVERYONE).strip().lower()
        if not PARTY.match(to):
            raise ValueError("'to' must be a party name, 'owner' or 'all'")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("'text' must be a non-empty string")
        if len(text) > max_chars:
            raise ValueError(f"'text' is longer than {max_chars} characters")
        with self._lock:
            message = {"id": self._last_id() + 1, "at": round(time.time(), 3),
                       "from": sender, "to": to, "text": text}
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as handle:
                handle.write(json.dumps(message) + "\n")
            messages = self._all()
            if len(messages) > keep * TRIM_MARGIN:
                temporary = self.path.with_suffix(".tmp")
                temporary.write_text("".join(json.dumps(m) + "\n" for m in messages[-keep:]))
                temporary.replace(self.path)
        return message

    def read(self, reader: str, after: int = 0) -> dict:
        """The messages after `after` that `reader` may see: to it, to all, or from it; the owner sees all."""
        messages = [m for m in self._all() if m["id"] > after]
        if reader != OWNER:
            messages = [m for m in messages if m["to"] in (reader, EVERYONE) or m["from"] == reader]
        return {"messages": messages, "last": messages[-1]["id"] if messages else after}
