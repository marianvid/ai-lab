"""Where the public services AI-Lab reads from live: PyPI, Hugging Face, GitHub.

The addresses are configuration, not code (golden rule 5), so they are kept in
`public_services.json` beside this module and read once. They are the same on
every machine, which is why they ship with AI-Lab rather than in a machine's
own configuration file. Each is a template: `{package}`, `{repo}`, `{path}`
and `{count}` are filled in by the caller.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path

ADDRESSES = Path(__file__).with_name("public_services.json")


@cache
def address(name: str) -> str:
    """The address template called `name`; KeyError naming the file when it is missing."""
    addresses = json.loads(ADDRESSES.read_text(encoding="utf-8"))
    if name not in addresses:
        raise KeyError(f"{ADDRESSES.name} has no address called {name!r}")
    return str(addresses[name])
