"""Handing a command line to a systemd unit.

systemd units are static files, but instances are created at runtime, so the
command cannot live in the unit. Instead the manager writes the command to a
small JSON file and a templated unit runs a launcher that reads it and execs.

    $AI_LAB_STATE_DIR/launch/<instance-id>.json
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

# The deployment says where the manager may write: both systemd units set this
# variable, so the path lives in the unit files, not in the code.
STATE_DIR_VARIABLE = "AI_LAB_STATE_DIR"
INSTANCE_ID = __import__("re").compile(r"^[a-z0-9][a-z0-9-]*$")


def state_dir() -> Path:
    """The writable directory the deployment gives the manager."""
    value = os.environ.get(STATE_DIR_VARIABLE, "").strip()
    if not value:
        raise RuntimeError(f"{STATE_DIR_VARIABLE} is not set; the systemd units set it")
    return Path(value)


def launch_dir() -> Path:
    return state_dir() / "launch"


def write_spec(spec: ProcessSpec, directory: Path | None = None) -> Path:
    if not INSTANCE_ID.match(spec.instance_id):
        raise ValueError(f"Invalid instance id: {spec.instance_id}")
    directory = directory or launch_dir()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{spec.instance_id}.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"argv": spec.argv, "env": spec.env}))
    temporary.replace(path)
    return path


def read_spec(instance_id: str, directory: Path | None = None) -> tuple[list[str], dict]:
    if not INSTANCE_ID.match(instance_id):
        raise ValueError(f"Invalid instance id: {instance_id}")
    directory = directory or launch_dir()
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
