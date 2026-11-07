"""Moving YuE2 songs between machines: through this machine's scheduler, never AI-Lab to AI-Lab.

The AI-Labs never talk to each other directly (the owner's decision, 2026-11-07). A song
on the other machine is reached through this machine's scheduler door, which passes the
call to the partner scheduler, which passes it to its own AI-Lab:

    {scheduler}/ai/<peer provider>/v1/audio/music/generations/<id>/bundle?model=<m>

Configured in the YuE2 engine's section as `transfer` (`engines.yue2.transfer`):

    "transfer": {
      "machine": "ai-lab-mac",
      "scheduler": "http://127.0.0.1:8230",
      "peers": ["ai-lab-linux"],
      "timeout_s": 60, "chunk_bytes": 1048576
    }

- `machine`: the name this AI-Lab goes by in a pull's answer.
- `scheduler`: this machine's scheduler, whose model door (`/ai/<provider>/...`) reaches
  the other machine. AI-Lab sends it no token: the door asks none of its callers.
- `peers`: the other machines' providers (their names in the LLM Map, as the door knows
  them) a song may be pulled from. Empty: no pull is possible.
- `timeout_s`: how long the door may stay silent while a song is fetched
  (seconds). Through the partner the answer only starts once the partner holds the whole
  song (~50 MB), so this covers the whole fetch on the other side.
- `chunk_bytes`: the piece a song is read, written and checked in (bytes).

Nothing here is a hidden value: the defaults below are what applies when the section
leaves a setting out, and `docs/music.md` lists them with the same explanations.
"""
from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from urllib.parse import urlparse

# The defaults a section may leave out (see the module's text for what each is).
DEFAULT_TIMEOUT_S = 60.0
DEFAULT_CHUNK_BYTES = 1 << 20
LONGEST_TIMEOUT_S = 3600.0
SMALLEST_CHUNK_BYTES = 4096
LARGEST_CHUNK_BYTES = 64 << 20
TRANSFER_FIELDS = {"machine", "scheduler", "peers", "timeout_s", "chunk_bytes"}
# A provider's name is one segment of the door's path.
PROVIDER_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


@dataclass(frozen=True)
class Peer:
    """Another machine, as this machine's scheduler door reaches it."""

    name: str
    url: str          # `{scheduler}/ai/<name>`: the other AI-Lab's paths follow it


@dataclass(frozen=True)
class TransferSettings:
    """`engines.yue2.transfer`, checked."""

    machine: str = ""
    scheduler: str = ""
    peers: dict[str, Peer] = field(default_factory=dict)
    timeout_s: float = DEFAULT_TIMEOUT_S
    chunk_bytes: int = DEFAULT_CHUNK_BYTES

    @classmethod
    def from_mapping(cls, raw: Mapping | None) -> TransferSettings:
        """Read and check the section; ValueError names what is wrong."""
        raw = {} if raw is None else raw
        if not isinstance(raw, Mapping):
            raise ValueError("yue2 transfer must be an object")
        unknown = set(raw) - TRANSFER_FIELDS
        if unknown:
            raise ValueError("Unknown yue2 transfer settings: " + ", ".join(sorted(unknown)))
        machine = raw.get("machine", "")
        if not isinstance(machine, str):
            raise ValueError("yue2 transfer machine must be text")
        scheduler = _scheduler(raw.get("scheduler", ""))
        names = _peer_names(raw.get("peers", []))
        if names and not scheduler:
            raise ValueError("yue2 transfer peers need the scheduler whose door reaches them")
        return cls(machine=machine, scheduler=scheduler,
                   peers={name: Peer(name, f"{scheduler}/ai/{name}") for name in names},
                   timeout_s=_timeout(raw.get("timeout_s", DEFAULT_TIMEOUT_S)),
                   chunk_bytes=_chunk(raw.get("chunk_bytes", DEFAULT_CHUNK_BYTES)))

    def peer(self, name) -> Peer:
        """The peer called `name`; ValueError when there is none."""
        found = self.peers.get(name) if isinstance(name, str) else None
        if found is None:
            known = ", ".join(sorted(self.peers)) or "none"
            raise ValueError(f"no peer {name!r}; configured: {known}")
        return found


def _scheduler(value) -> str:
    if value == "":
        return ""
    if not isinstance(value, str) or urlparse(value).scheme not in ("http", "https") \
            or not urlparse(value).netloc:
        raise ValueError("yue2 transfer scheduler must be http(s)://host:port")
    return value.rstrip("/")


def _peer_names(value) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(name, str) for name in value):
        raise ValueError("yue2 transfer peers must be a list of provider names")
    for name in value:
        if not PROVIDER_NAME.fullmatch(name):
            raise ValueError(f"yue2 transfer peer {name!r} is not a provider name")
    if len(set(value)) != len(value):
        raise ValueError("yue2 transfer peers name a provider twice")
    return list(value)


def _timeout(value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value <= LONGEST_TIMEOUT_S:
        raise ValueError(f"yue2 transfer timeout_s must be above 0 and at most {LONGEST_TIMEOUT_S}")
    return float(value)


def _chunk(value) -> int:
    if type(value) is not int or not SMALLEST_CHUNK_BYTES <= value <= LARGEST_CHUNK_BYTES:
        raise ValueError(f"yue2 transfer chunk_bytes must be a whole number from "
                         f"{SMALLEST_CHUNK_BYTES} to {LARGEST_CHUNK_BYTES}")
    return value
