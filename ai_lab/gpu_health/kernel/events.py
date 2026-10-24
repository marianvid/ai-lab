"""The kernel's messages about the card, as the helper on the Proxmox host writes them.

The container cannot read the kernel log (`dmesg` is refused even to root
there), so a small helper on the host follows it and writes one JSON line per
interesting message into a file the container sees:

    {"ts": 1761300000.0, "kind": "kernel", "line": "NVRM: Xid (PCI:0000:01:00): 79, ..."}
    {"ts": 1761300000.0, "kind": "heartbeat"}

The heartbeat comes every hour, so silence can be told apart from "nothing
happened". This module reads the lines added since last time (it remembers
where it stopped, and starts over when the helper rotated the file) and turns
them into alarms: an Xid (the driver's error code) and a PCIe AER error.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from ..alarms import Condition
from ..settings import GpuHealthSettings

XID = re.compile(r"NVRM: Xid \(PCI:([0-9a-fA-F:.]+)\): (\d+)(?:, (.*))?")
# What one read may take, so a huge backlog is read over several turns.
CHUNK = 1 << 20
# "0000:01:00.0" has a PCI domain ("0000:") in front of the bus address.
WITH_DOMAIN = 2


class KernelEvents:
    """Follows the helper's file; `position` (inode, offset, last heartbeat) survives restarts."""

    def __init__(self, settings: GpuHealthSettings, position: dict | None = None) -> None:
        self._settings = settings
        self._path = Path(settings.helper_events_path) if settings.helper_events_path else None
        self.position = {"inode": None, "offset": 0, "heartbeat": None, **(position or {})}

    @property
    def installed(self) -> bool:
        """Whether a helper is configured at all."""
        return self._path is not None

    def read(self) -> list[Condition]:
        """The alarms in the lines added since the last read.

        When the helper rotated the file, the end of the old one (now
        `events.jsonl.1`) is read first, so nothing written just before the
        rotation is lost.
        """
        if self._path is None or not self._path.is_file():
            return []
        stat = self._path.stat()
        data = b""
        if stat.st_ino != self.position["inode"]:
            data = self._rest_of_rotated(self._path)
            self.position.update(inode=stat.st_ino, offset=0)
        elif stat.st_size < self.position["offset"]:
            self.position["offset"] = 0
        complete = _complete_lines(self._path, self.position["offset"])
        self.position["offset"] += len(complete)
        found = []
        for raw in (data + complete).decode("utf-8", "replace").splitlines():
            found += self._line(raw)
        return found

    def _rest_of_rotated(self, path: Path) -> bytes:
        """What the old file got after the last read, if it is the one we were reading."""
        old = path.with_name(path.name + ".1")
        if self.position["inode"] is None or not old.is_file() \
                or old.stat().st_ino != self.position["inode"]:
            return b""
        return _complete_lines(old, self.position["offset"])

    def _line(self, raw: str) -> list[Condition]:
        try:
            entry = json.loads(raw)
        except ValueError:
            return []
        if not isinstance(entry, dict):
            return []
        if entry.get("kind") == "heartbeat":
            self.position["heartbeat"] = entry.get("ts")
            return []
        return parse(str(entry.get("line", "")), self._settings)

    def silent(self, now: float) -> list[Condition]:
        """The helper's heartbeat missing for longer than `helper_heartbeat_s`."""
        if self._path is None:
            return []
        last = self.position["heartbeat"]
        if last is not None and now - last <= self._settings.helper_heartbeat_s:
            return []
        when = "never heard from" if last is None else f"silent for {(now - last) / 3600:.1f} h"
        return [Condition("helper_silent", "warning",
                          f"The kernel-log helper on the host is {when}: Xid and AER "
                          f"errors are not being watched", {"last": last,
                                                           "path": os.fspath(self._path)})]


def _complete_lines(path: Path, offset: int) -> bytes:
    """The whole lines of `path` from `offset` on (a line still being written waits)."""
    with path.open("rb") as handle:
        handle.seek(offset)
        data = handle.read(CHUNK)
    return data[:data.rfind(b"\n") + 1]


def parse(line: str, settings: GpuHealthSettings) -> list[Condition]:
    """An Xid or an AER error in one kernel line; [] for anything else."""
    match = XID.search(line)
    if match:
        code = int(match.group(2))
        level = "critical" if code in settings.xid_critical else "warning"
        return [Condition(f"xid:{code}", level, f"Xid {code} from the driver: "
                          f"{(match.group(3) or '').strip() or 'no detail'}",
                          {"xid": code, "pci": match.group(1), "line": line})]
    if "AER" not in line or not _names_card(line, settings.pci_address):
        return []
    if "Uncorrect" in line or "Fatal" in line:
        return [Condition("aer_uncorrectable", "critical",
                          "PCIe AER: uncorrectable error on the card's link", {"line": line})]
    return [Condition("aer_corrected", "info",
                      "PCIe AER: corrected error on the card's link", {"line": line})]


def _names_card(line: str, address: str) -> bool:
    """Whether the line names the card (with or without the PCI domain "0000:")."""
    if not address:
        return True
    short = address.split(":", 1)[1] if address.count(":") == WITH_DOMAIN else address
    return address in line or short in line
