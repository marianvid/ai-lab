"""The checks every music engine makes on a request, written once.

Each engine (`*_backend.py`) accepts its own fields with its own limits and
words, but the checks are the same kinds: no unknown fields, the right model,
a prompt and lyrics of bounded length, a duration in a range, a 32-bit seed,
an optional source recording sent as base64 WAV. The answer has one shape
too: base64 WAV under `data`, with what the engine adds beside it.
"""

from __future__ import annotations

import base64
import binascii
import secrets
import wave
from dataclasses import dataclass
from pathlib import Path

# What one request may carry, unless an engine says otherwise.
MAX_PROMPT_CHARS = 4000
MAX_LYRICS_CHARS = 12000
# A seed is a 32-bit unsigned integer.
SEED_VALUES = 2**32
MAX_SEED = SEED_VALUES - 1


def check_fields(body: dict, allowed: set[str], engine: str, model_name: str) -> None:
    """Refuse fields this engine does not know, and a model it does not serve."""
    unknown = set(body) - allowed
    if unknown:
        raise ValueError(f"Unknown {engine} fields: " + ", ".join(sorted(unknown)))
    if body.get("model") not in (None, model_name):
        raise ValueError(f"this engine serves {model_name}")


@dataclass(frozen=True)
class TextRule:
    """How long one text field may be, and what to say when it is not.

    `shortest` and `longest` count the characters left after trimming the
    ends. `coerce` turns any value into text (`str(value)`); without it a
    value that is not text is refused like one of the wrong length.
    """

    message: str
    longest: int
    shortest: int = 0
    coerce: bool = False


def text(body: dict, field: str, rule: TextRule, default: str = "") -> str:
    """One text field, trimmed, within `rule`'s length; ValueError with its message."""
    value = body.get(field, default)
    if rule.coerce:
        value = str(value)
    if not isinstance(value, str) or not rule.shortest <= len(value.strip()) <= rule.longest:
        raise ValueError(rule.message)
    return value.strip()


def flag(body: dict, field: str, default: bool) -> bool:
    """A true-or-false field; anything else is refused."""
    value = body.get(field, default)
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be true or false")
    return value


def seconds(body: dict, default: float, shortest: float, longest: float) -> float:
    """The `duration` field: a number of seconds within the engine's range."""
    duration = body.get("duration", default)
    if type(duration) not in (int, float) or not shortest <= duration <= longest:
        raise ValueError(f"duration must be between {shortest} and {longest} seconds")
    return duration


def seed(body: dict, drawn_below: int = SEED_VALUES) -> int:
    """The `seed` field, or a random one below `drawn_below` when it is left out."""
    value = body.get("seed", secrets.randbelow(drawn_below))
    if type(value) is not int or not 0 <= value <= MAX_SEED:
        raise ValueError("seed must be a 32-bit unsigned integer")
    return value


@dataclass(frozen=True)
class WavRule:
    """The largest source recording an engine takes, and its words for each refusal."""

    limit_bytes: int
    missing_or_large: str
    not_base64: str
    not_wav: str


def decoded_wav(encoded, rule: WavRule) -> bytes:
    """A base64 WAV from the request, checked for size and header; not decoded to samples."""
    if not isinstance(encoded, str) or not encoded or \
            len(encoded) > rule.limit_bytes * 4 // 3 + 4:
        raise ValueError(rule.missing_or_large)
    try:
        audio = base64.b64decode(encoded, validate=True)
    except binascii.Error:
        raise ValueError(rule.not_base64) from None
    if not audio.startswith(b"RIFF") or len(audio) > rule.limit_bytes:
        raise ValueError(rule.not_wav)
    return audio


def wav_length_s(path: Path) -> float:
    """How long a WAV file plays, in seconds."""
    with wave.open(str(path)) as audio:
        return audio.getnframes() / audio.getframerate()


def wav_answer(audio: bytes, **fields) -> dict:
    """The music answer: `fields` beside the WAV, base64 under `data`."""
    return {**fields, "data": [{"mime_type": "audio/wav",
                                "b64_wav": base64.b64encode(audio).decode()}]}


def checked_wav(audio: bytes, engine: str) -> bytes:
    """The engine's output, refused unless it is a WAV file."""
    if not audio.startswith(b"RIFF"):
        raise RuntimeError(f"{engine} did not return WAV audio")
    return audio
