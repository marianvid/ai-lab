"""The two addresses that are the same on every machine, named once.

- `LOOPBACK` — this machine only. AI-Lab reaches its own engines here.
- `ALL_INTERFACES` — listen on every network card, so other machines on the
  network can reach a page or an engine.

They are not machine-specific values (CONVENTIONS rule 5): every computer has
them. They are written here once and used by name everywhere else.
"""

LOOPBACK = "127.0.0.1"
ALL_INTERFACES = "0.0.0.0"


def local_url(port: int, path: str = "") -> str:
    """The URL of something listening on this machine, e.g. an engine."""
    return f"http://{LOOPBACK}:{port}{path}"
