"""What a CLI's failure means, read from its own words.

A CLI reports everything as an exit code plus prose, so the only way to tell a
rate limit from a lost login is the text. Four classes, each with its own
answer:

- **login**: the vendor says its sign-in is dead (an expired login, a token it
  refuses). Retrying cannot help and the caller must be told plainly: answer
  at once with its own error (`ProviderLoginExpiredError`, 503
  `vendor_login_expired`). Checked first, and only on the vendor's own words.
- **rate limit**: the vendor's allowance is spent for now. Wait, then retry
  the same model; the whole vendor waits with it.
- **fatal**: retrying cannot help (not signed in, unknown model). Answer at
  once, so the caller's own fallback takes over.
- **transient**: anything else, a timeout or a lost connection. Retry a few
  times. Text that matches nothing is transient but keeps its own label, so a
  vendor rewording its limit message shows up as a new, growing counter.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any

# Long enough to recognise a message again, short enough for a counter name.
LABEL_LENGTH = 60


class ErrorKind(Enum):
    """The four ways a CLI call can fail."""

    LOGIN = "login"
    RATE_LIMIT = "rate_limit"
    FATAL = "fatal"
    TRANSIENT = "transient"


# The vendors' own words for a dead sign-in. Only phrases a vendor uses for its
# login, never a bare "401" or "unauthorized": a false "expired" would send
# every caller to its fallback. Where each was seen:
LOGIN_MARKERS = (
    # Claude Code, a browser login that ended (CT 102, 2026-11-14):
    # "Failed to authenticate: OAuth session expired and could not be refreshed".
    "oauth session expired", "could not be refreshed",
    # Claude Code, a yearly token it refuses (made-up token, Mac, 2026-11-17):
    # "Failed to authenticate. API Error: 401 Invalid bearer token".
    "failed to authenticate", "invalid bearer token",
    # Claude Code, no sign-in at all (Mac, 2026-11-01 and 2026-11-17):
    # "Not logged in · Please run /login"; Anthropic's docs: "Login expired".
    "not logged in", "please run /login", "login expired", "oauth token has expired",
    # Codex, no or unusable auth.json (Mac, 2026-11-17): "401 Unauthorized:
    # Missing bearer or basic authentication in header".
    "missing bearer or basic authentication",
    # Codex, a ChatGPT login whose refresh failed (not seen live yet).
    "please log out and sign in again",
)
LOGIN_LABEL = "login expired"
RATE_LIMIT_MARKERS = (
    "credits out", "usage limit", "rate limit", "429", "too many requests",
    "quota", "try again later", "resets at", "resets in", "overloaded",
    "capacity", "session limit",
)
# Of those, the ones that mean the subscription's allowance (hours, a week),
# not a short refusal of a few seconds: the owner must know of these at once.
USAGE_LIMIT_MARKERS = (
    "credits out", "usage limit", "quota", "resets at", "resets in", "session limit",
)
USAGE = "usage"
RATE = "rate"


def limit_kind(label: str) -> str:
    """`usage` for the subscription's allowance, `rate` for a short refusal."""
    return USAGE if label in USAGE_LIMIT_MARKERS else RATE


FATAL_MARKERS = (
    "unauthorized", "invalid api key", "no such model",
    "unknown model", "permission denied", "forbidden",
    "cannot start the cli",
    # Codex, asked for a model the subscription does not include (seen live
    # with gpt-5.6-sol on a ChatGPT account): retrying only wastes 20 s.
    "is not supported", "invalid_request_error",
)


def classify(text: str) -> tuple[ErrorKind, str]:
    """The class of a failure, and the label it is counted under."""
    lowered = (text or "").lower()
    if any(marker in lowered for marker in LOGIN_MARKERS):
        return ErrorKind.LOGIN, LOGIN_LABEL
    for marker in RATE_LIMIT_MARKERS:
        if marker in lowered:
            return ErrorKind.RATE_LIMIT, marker
    for marker in FATAL_MARKERS:
        if marker in lowered:
            return ErrorKind.FATAL, marker
    return ErrorKind.TRANSIENT, _label(lowered)


def _label(lowered: str) -> str:
    """A stable name for an unrecognised message: its last line, digits blanked."""
    lines = [line.strip() for line in lowered.splitlines() if line.strip()]
    if not lines:
        return "empty output"
    return re.sub(r"\d+", "#", lines[-1])[:LABEL_LENGTH]


class ProviderError(Exception):
    """A subscription model could not answer. `kind` is the OpenAI error type."""

    kind = "api_error"


class ProviderRateLimitError(ProviderError):
    """The vendor's allowance stayed spent through every retry."""

    kind = "rate_limit_error"


class ProviderUnavailableError(ProviderError):
    """Signed out, unknown model, or a failure that outlasted its retries."""


class ProviderVendorLimitedError(ProviderError):
    """A call that asked not to be held (`"ai_lab": {"hold": false}`) met a limited vendor.

    Answered at once, so the caller's own fallback (a local model) takes over
    instead of waiting for the vendor's pause. 503 `vendor_limited`, with the
    vendor, the machine, the kind of limit and the times AI-Lab knows.
    """

    kind = "rate_limit_error"
    code = "vendor_limited"

    def __init__(self, vendor: str, machine: str, limit: dict) -> None:
        where = machine or "this machine"
        words = limit.get("said") or "no words from the vendor"
        what = "usage limit" if limit.get("kind") == USAGE else "rate limit"
        super().__init__(f"{vendor} is at its {what} on {where}: {words}")
        about: dict[str, str] = {
            "vendor": vendor, "machine": machine, "kind": limit.get("kind", ""),
            "retry_at": limit.get("retry_at", ""), "resets_at": limit.get("resets_at", "")}
        self.detail: dict[str, dict[str, Any]] = {
            "ai_lab": {"reason": "vendor_limited", "detail": about}}


class ProviderCallCancelledError(ConnectionAbortedError):
    """The client left before its answer was ready; its CLI was ended, or never started.

    Not a `ProviderError`: nobody is left to answer, so the front door closes
    the connection and says nothing (as for a local call whose client left).
    Not a refusal either: the vendor's pause never hears of it.
    """


class ProviderLoginExpiredError(ProviderError):
    """The vendor said its sign-in on this machine is dead: no retry helps.

    `detail` names the vendor and the machine, so a caller (the scheduler,
    a team's run, the console) can say which login to renew.
    """

    kind = "authentication_error"
    code = "vendor_login_expired"

    def __init__(self, vendor: str, machine: str, said: str) -> None:
        where = machine or "this machine"
        super().__init__(f"The {vendor} login on {where} has expired: {said}")
        self.detail = {"vendor": vendor, "machine": machine}


# A token-shaped run of text: an Anthropic key or token, or any long run of
# letters and digits. Blanked before a CLI's words leave AI-Lab.
_SECRET = re.compile(r"sk-[A-Za-z0-9_-]{8,}|[A-Za-z0-9_-]{32,}")
SAID_LENGTH = 200


def said(text: str) -> str:
    """The CLI's words on one line, short, with anything token-shaped blanked."""
    flat = " ".join((text or "").split())
    return _SECRET.sub("[…]", flat)[-SAID_LENGTH:] or "no output"
