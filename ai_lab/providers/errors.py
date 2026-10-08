"""What a CLI's failure means, read from its own words.

A CLI reports everything as an exit code plus prose, so the only way to tell a
rate limit from a lost login is the text. Three classes, each with its own
answer:

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

# Long enough to recognise a message again, short enough for a counter name.
LABEL_LENGTH = 60


class ErrorKind(Enum):
    """The three ways a CLI call can fail."""

    RATE_LIMIT = "rate_limit"
    FATAL = "fatal"
    TRANSIENT = "transient"


RATE_LIMIT_MARKERS = (
    "credits out", "usage limit", "rate limit", "429", "too many requests",
    "quota", "try again later", "resets at", "resets in", "overloaded",
    "capacity", "session limit",
)
FATAL_MARKERS = (
    "not logged in", "unauthorized", "invalid api key", "no such model",
    "unknown model", "permission denied", "forbidden", "please run /login",
    "cannot start the cli",
    # Codex, asked for a model the subscription does not include (seen live
    # with gpt-5.6-sol on a ChatGPT account): retrying only wastes 20 s.
    "is not supported", "invalid_request_error",
)


def classify(text: str) -> tuple[ErrorKind, str]:
    """The class of a failure, and the label it is counted under."""
    lowered = (text or "").lower()
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
