"""The addresses AI-Lab listens on, named once.

- `LOOPBACK` — this machine only. Every engine's API listens here, and AI-Lab
  reaches its own engines here. Other machines reach the models only through
  AI-Lab's own port, which asks for a token.
- `ALL_INTERFACES` — every network card. AI-Lab's own port listens here.
- `PAGE_INTERFACES` — where the pages a person opens from an engine's button
  (Music, Speak, Create, Edit) listen: ComfyUI, the studios, the editors.
  Today every network card, so those buttons work from another computer.

They are not machine-specific values (CONVENTIONS rule 5): every computer has
them. They are written here once and used by name everywhere else.
"""

LOOPBACK = "127.0.0.1"
ALL_INTERFACES = "0.0.0.0"
PAGE_INTERFACES = ALL_INTERFACES


def local_url(port: int, path: str = "") -> str:
    """The URL of something listening on this machine, e.g. an engine."""
    return f"http://{LOOPBACK}:{port}{path}"
