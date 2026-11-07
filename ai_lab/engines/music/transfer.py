"""Moving YuE2 songs between machines: who the other AI-Labs are, and how a song travels.

Configured in the YuE2 engine's section as `transfer` (`engines.yue2.transfer`):

    "transfer": {
      "machine": "ai-lab-mac",
      "peers": {"ai-lab-linux": {"url": "http://<address>:8090",
                                 "token_file": "/path/in/opts/client-tokens.json",
                                 "token_key": "<client name>"}},
      "timeout_s": 60, "chunk_bytes": 1048576
    }

- `machine`: the name this AI-Lab goes by in a pull's answer.
- `peers`: each other AI-Lab, by name: its address, and where its client token is kept.
  The token is never in the configuration: `token_file` names a JSON file kept privately
  (in `opts`), and `token_key` the entry in it. It is read when a pull runs.
- `timeout_s`: how long a peer may stay silent while a song is fetched or deleted
  (seconds; a whole song of ~50 MB is fetched in pieces, each waited for this long).
- `chunk_bytes`: the piece a song is read, written and checked in (bytes).

Nothing here is a hidden value: the defaults below are what applies when the section
leaves a setting out, and `docs/music.md` lists them with the same explanations.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

# The defaults a section may leave out (see the module's text for what each is).
DEFAULT_TIMEOUT_S = 60.0
DEFAULT_CHUNK_BYTES = 1 << 20
LONGEST_TIMEOUT_S = 3600.0
SMALLEST_CHUNK_BYTES = 4096
LARGEST_CHUNK_BYTES = 64 << 20
PEER_FIELDS = {"url", "token_file", "token_key"}
TRANSFER_FIELDS = {"machine", "peers", "timeout_s", "chunk_bytes"}


@dataclass(frozen=True)
class Peer:
    """Another AI-Lab: its address, and where its client token is kept."""

    name: str
    url: str
    token_file: str
    token_key: str

    def token(self) -> str:
        """The client token, read from its private file when it is needed."""
        try:
            tokens = json.loads(Path(self.token_file).read_text())
        except (OSError, ValueError) as error:
            raise RuntimeError(f"the token file for peer {self.name} cannot be read: {error}") from None
        value = tokens.get(self.token_key) if isinstance(tokens, dict) else None
        if not isinstance(value, str) or not value:
            raise RuntimeError(f"the token file for peer {self.name} has no {self.token_key!r}")
        return value


@dataclass(frozen=True)
class TransferSettings:
    """`engines.yue2.transfer`, checked."""

    machine: str = ""
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
        peers = raw.get("peers", {})
        if not isinstance(peers, Mapping):
            raise ValueError("yue2 transfer peers must be an object")
        return cls(machine=machine, peers={name: _peer(name, value) for name, value in peers.items()},
                   timeout_s=_timeout(raw.get("timeout_s", DEFAULT_TIMEOUT_S)),
                   chunk_bytes=_chunk(raw.get("chunk_bytes", DEFAULT_CHUNK_BYTES)))

    def peer(self, name) -> Peer:
        """The peer called `name`; ValueError when there is none."""
        found = self.peers.get(name) if isinstance(name, str) else None
        if found is None:
            known = ", ".join(sorted(self.peers)) or "none"
            raise ValueError(f"no peer {name!r}; configured: {known}")
        return found


def _timeout(value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value <= LONGEST_TIMEOUT_S:
        raise ValueError(f"yue2 transfer timeout_s must be above 0 and at most {LONGEST_TIMEOUT_S}")
    return float(value)


def _chunk(value) -> int:
    if type(value) is not int or not SMALLEST_CHUNK_BYTES <= value <= LARGEST_CHUNK_BYTES:
        raise ValueError(f"yue2 transfer chunk_bytes must be a whole number from "
                         f"{SMALLEST_CHUNK_BYTES} to {LARGEST_CHUNK_BYTES}")
    return value


def _peer(name: str, value) -> Peer:
    if not isinstance(value, Mapping) or set(value) != PEER_FIELDS:
        raise ValueError(f"yue2 transfer peer {name} needs exactly: " + ", ".join(sorted(PEER_FIELDS)))
    url = value["url"]
    if not isinstance(url, str) or urlparse(url).scheme not in ("http", "https") or not urlparse(url).netloc:
        raise ValueError(f"yue2 transfer peer {name} url must be http(s)://host:port")
    token_file, token_key = value["token_file"], value["token_key"]
    if not isinstance(token_file, str) or not Path(token_file).is_absolute():
        raise ValueError(f"yue2 transfer peer {name} token_file must be an absolute path")
    if not isinstance(token_key, str) or not token_key:
        raise ValueError(f"yue2 transfer peer {name} token_key must name the token's entry")
    return Peer(name, url.rstrip("/"), token_file, token_key)
