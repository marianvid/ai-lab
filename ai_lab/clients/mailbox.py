"""A mailbox for agents that talk to each other through AI-Lab, and the owner who reads them.

Asked by the owner (2026-11-23) so that the ai-lab agent and the sqwad.ai agent
on his VM can write to each other without him carrying every message. The
rules came from the sqwad agent and he agreed:

- a message is information, not a command: nothing here executes anything;
- the owner can read the whole history (the web interface reads every message);
- the mailbox has its own tokens, separate from any token that calls a model.

A party is a registered client named `mailbox-<party>`: `mailbox-sqwad` writes
as `sqwad`. Such a client opens the mailbox only (the door refuses it anywhere
else). The web interface writes as `owner`.

Messages are kept one JSON object per line, appended, the oldest dropped past
`keep`: a mailbox is a conversation, not an archive.
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
KEEP = 2000
MAX_CHARS = 20000
PARTY = re.compile(r"^[a-z0-9][a-z0-9-]{0,31}$")


def party_of(client: str | None) -> str | None:
    """The mailbox party behind a client name, or None for a client that is not one."""
    if client and client.startswith(PREFIX) and PARTY.match(client[len(PREFIX):]):
        return client[len(PREFIX):]
    return None


class Mailbox:
    """Messages between parties, read by `after` id."""

    def __init__(self, path: Path, keep: int = KEEP, max_chars: int = MAX_CHARS) -> None:
        self.path = Path(path)
        self.keep = keep
        self.max_chars = max_chars
        self._lock = threading.Lock()

    def _all(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]

    def post(self, sender: str, to: str, text: str) -> dict:
        """Keep one message; its id and time are returned with it."""
        to = (to or EVERYONE).strip().lower()
        if not PARTY.match(to) and to not in (EVERYONE, OWNER):
            raise ValueError("'to' must be a party name, 'owner' or 'all'")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("'text' must be a non-empty string")
        if len(text) > self.max_chars:
            raise ValueError(f"'text' is longer than {self.max_chars} characters")
        with self._lock:
            messages = self._all()
            last = messages[-1]["id"] if messages else 0
            message = {"id": last + 1, "at": round(time.time(), 3), "from": sender, "to": to, "text": text}
            messages.append(message)
            messages = messages[-self.keep:]
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix(".tmp")
            temporary.write_text("".join(json.dumps(m) + "\n" for m in messages))
            temporary.replace(self.path)
        return message

    def read(self, reader: str, after: int = 0) -> dict:
        """The messages after `after` that `reader` may see: to it, to all, or from it; the owner sees all."""
        messages = [m for m in self._all() if m["id"] > after]
        if reader != OWNER:
            messages = [m for m in messages if m["to"] in (reader, EVERYONE) or m["from"] == reader]
        return {"messages": messages, "last": messages[-1]["id"] if messages else after}
