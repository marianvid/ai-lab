"""What the isolated engine processes share: the small HTTP hosts they answer on.

Every engine server script (`audio/`, `speech/`, `music/`, `images/`, `text/`)
runs in the engine's own Python environment, started by the engine's command
line, never imported by the manager. This package is the only part of AI-Lab
built for that side besides those scripts; it uses the standard library and
`network` only. The manager's own media job queues are in `media/`.
"""
