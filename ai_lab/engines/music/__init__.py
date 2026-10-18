"""Engines for music models (one module per engine; found by `engines.registry`)."""


def music_server(engine: str) -> tuple[str, ...]:
    """Where an engine's music host sits, as parts of a path under `ai_lab/`."""
    return ("music", "servers", f"{engine}_server.py")
