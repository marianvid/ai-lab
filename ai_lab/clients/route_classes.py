"""Which routes a registered client's token reaches: the model API, and nothing else.

Everything else is management: Security, the clients and their tokens, the
settings, the entries and their loads, the engines' installs and builds, the
downloads, the library, storage, the switch, the journal, the card's health.
Those are for the page: its session when login is off, a logged-in one when it
is on (radio-lab, `2026-11-27-ai-lab-admin-routes-need-a-login`). The list of
every route, its class and who calls it is in that request's `ai-lab.md`.
"""

from __future__ import annotations

# Every model call, model listing and song file is under /v1/.
MODEL_PREFIX = "/v1/"
# Outside /v1/: counting tokens, the caller's own heartbeat and mailbox, and the
# gateway's state a client reads (the queue, the logins, the usage limits).
MODEL_ROUTES = frozenset({("POST", "/tokenize"), ("POST", "/api/clients/heartbeat"),
                          ("GET", "/api/mailbox"), ("POST", "/api/mailbox"),
                          ("GET", "/api/gateway")})


def model_api(method: str, path: str) -> bool:
    """Whether a client's token may call this route; False means it is for the page."""
    return path.startswith(MODEL_PREFIX) or (method.upper(), path) in MODEL_ROUTES
