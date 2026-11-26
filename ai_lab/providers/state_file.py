"""A small JSON file of per-vendor states, beside the usage counters.

Read once at start; written whole, through a spare file, when a state
changes. A file that cannot be read starts empty; one that cannot be written
is skipped (the page then only forgets it at the next restart).
"""

from __future__ import annotations

import json
from pathlib import Path


def read(path: Path) -> dict[str, dict]:
    """The states kept in `path`, or none."""
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {key: value for key, value in data.items() if isinstance(value, dict)}


def write(path: Path, states: dict[str, dict]) -> None:
    """Replace `path` with `states`, never leaving half a file."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        spare = path.with_suffix(".tmp")
        spare.write_text(json.dumps(states, indent=1, sort_keys=True))
        spare.replace(path)
    except OSError:
        pass
