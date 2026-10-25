#!/usr/bin/env python3
"""The GPU health monitor's helper on the Proxmox host: the kernel's messages about the card.

It runs on the host, not in AI-Lab: a copy of this one file, started by
`system/gpu-health/ai-lab-gpu-kernel-log.service` with the folder to write
into as its only argument. It uses the Python standard library only and
imports nothing from AI-Lab, so the copy works on its own.

AI-Lab runs in a container that may not read the kernel log, so this small
program runs on the host instead. It follows the kernel log (`journalctl -k`)
and writes one JSON line per message that matters into a file the container
can read through a read-only bind mount:

    {"ts": 1761300000.0, "kind": "kernel", "line": "NVRM: Xid (PCI:0000:01:00): 79, ..."}
    {"ts": 1761300000.0, "kind": "heartbeat"}

"Messages that matter": the NVIDIA driver's Xid errors and PCIe AER errors.
AI-Lab decides which of them concern the card. A heartbeat is written every
hour, so the monitor can tell "nothing happened" from "the helper is dead".

Where it resumes: journalctl's cursor file, so nothing is lost across a
restart of this helper; the first start begins from now. The file is rotated
at 10 MB (one old copy kept). Python 3 standard library only.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

INTERESTING = re.compile(r"NVRM: Xid|AER")
MAX_BYTES = 10 * 1024 * 1024
HEARTBEAT_S = 3600.0


class Output:
    """The events file: appended under a lock, readable by everyone, rotated when large."""

    def __init__(self, folder: Path) -> None:
        self.path = folder / "events.jsonl"
        self._lock = threading.Lock()
        folder.mkdir(parents=True, exist_ok=True)
        folder.chmod(0o755)

    def write(self, entry: dict) -> None:
        """One JSON line."""
        with self._lock:
            if self.path.exists() and self.path.stat().st_size > MAX_BYTES:
                os.replace(self.path, self.path.with_suffix(".jsonl.1"))
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry) + "\n")
            self.path.chmod(0o644)


def message_of(entry: dict) -> str:
    """journalctl's MESSAGE: text, or a list of bytes when it was not valid UTF-8."""
    message = entry.get("MESSAGE", "")
    if isinstance(message, list):
        return bytes(message).decode("utf-8", "replace")
    return str(message)


def kernel_entry(raw: str) -> dict | None:
    """The line to write for one `journalctl -o json` line, or None when it does not matter."""
    try:
        entry = json.loads(raw)
    except ValueError:
        return None
    if not isinstance(entry, dict):
        return None
    message = message_of(entry)
    if not INTERESTING.search(message):
        return None
    stamp = int(entry.get("__REALTIME_TIMESTAMP", 0)) / 1e6 or time.time()
    return {"ts": stamp, "kind": "kernel", "line": message}


def command(folder: Path) -> list[str]:
    """journalctl following the kernel log from where it stopped last time.

    The first time there is no cursor file yet, so it starts from now:
    `--lines=0` (no old lines). Not `--since=now`: journalctl refuses
    `--since` together with `--cursor-file` (systemd 257 on the Proxmox host,
    "Please specify only one of --since=, --cursor=, --cursor-file=").
    """
    cursor = folder / "journal.cursor"
    arguments = ["journalctl", "-k", "-f", "-o", "json", f"--cursor-file={cursor}"]
    if not cursor.exists():
        arguments.append("--lines=0")
    return arguments


def beat(output: Output, stop: threading.Event, every: float = HEARTBEAT_S) -> None:
    """A heartbeat now and then every `every` seconds, until `stop`."""
    while not stop.is_set():
        output.write({"ts": time.time(), "kind": "heartbeat"})
        stop.wait(every)


def main(argv: list[str] | None = None) -> int:
    """Follow the kernel log until journalctl ends (systemd starts it again); 2: no folder."""
    arguments = sys.argv[1:] if argv is None else argv
    if len(arguments) != 1:
        print("usage: host_helper.py <folder to write events.jsonl into>", file=sys.stderr)
        return 2
    folder = Path(arguments[0])
    output = Output(folder)
    stop = threading.Event()
    threading.Thread(target=beat, args=(output, stop), daemon=True).start()
    with subprocess.Popen(command(folder), stdout=subprocess.PIPE, text=True) as journal:
        assert journal.stdout is not None
        for raw in journal.stdout:
            entry = kernel_entry(raw)
            if entry is not None:
                output.write(entry)
    stop.set()
    return 1


if __name__ == "__main__":
    sys.exit(main())
