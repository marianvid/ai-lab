"""Handing a command line to a systemd unit.

systemd units are static files, but instances are created at runtime, so the
command cannot live in the unit. Instead the manager writes the command to a
small JSON file and a templated unit runs a launcher that reads it and execs.

    /var/lib/ai-lab/launch/<instance-id>.json
    {"argv": ["llama-server", "--model", "..."], "env": {}}

The launcher deliberately understands nothing: no configuration parsing, no
engine logic, just read and exec. It runs as the same user as the manager, so
a manager-written command line grants no privilege the manager did not already
have.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from ..types import ProcessSpec

LAUNCH_DIR = Path("/var/lib/ai-lab/launch")
INSTANCE_ID = __import__("re").compile(r"^[a-z0-9][a-z0-9-]*$")


def write_spec(spec: ProcessSpec, directory: Path = LAUNCH_DIR) -> Path:
    if not INSTANCE_ID.match(spec.instance_id):
        raise ValueError(f"Invalid instance id: {spec.instance_id}")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{spec.instance_id}.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"argv": spec.argv, "env": spec.env}))
    temporary.replace(path)
    return path


def read_spec(instance_id: str, directory: Path = LAUNCH_DIR) -> tuple[list[str], dict]:
    if not INSTANCE_ID.match(instance_id):
        raise ValueError(f"Invalid instance id: {instance_id}")
    payload = json.loads((directory / f"{instance_id}.json").read_text())
    return payload["argv"], payload.get("env", {})


def main() -> None:
    """Entry point for `ai-lab-run <instance-id>`, called by the systemd unit."""
    if len(sys.argv) != 2:
        raise SystemExit("usage: ai-lab-run <instance-id>")
    argv, env = read_spec(sys.argv[1])
    os.execvpe(argv[0], argv, {**os.environ, **env})


if __name__ == "__main__":
    main()
