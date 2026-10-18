"""Request validation and WAV response shared by speech engines."""
from __future__ import annotations

import base64
import binascii
import contextlib
import io
from typing import Any

PATH = "/v1/audio/speech/generations"

# A reference voice is a short clip; thirty seconds of 48 kHz stereo PCM16 is
# under 6 MB. The limit keeps one request from carrying an album.
MAX_REFERENCE_BYTES = 8 * 1024 * 1024
MAX_SEED = 2**31 - 1
# A request body: the reference in base64 (a third larger) plus the text.
MAX_REQUEST_BYTES = 12 * 1024 * 1024
# The longest text, and reference transcript, one request may carry.
MAX_TEXT_CHARS = 2000
MAX_INSTRUCTION_CHARS = 1000
MAX_LANGUAGE_CHARS = 32
MAX_SPEAKER_CHARS = 80


def validate_payload(body: dict, *, seed: bool = False,
                     reference: bool = False) -> dict[str, Any]:
    """Check a speech request against what this engine can honour.

    `seed` and `reference` say whether the calling engine uses those fields.
    An engine that does not is sent a request carrying them only by mistake,
    and ignoring the field would hand back a different voice than the one
    asked for without a word — so it is refused by name instead.
    """
    _refuse_unknown(body, seed=seed, reference=reference)
    text = str(body.get("text", "")).strip()
    if not 1 <= len(text) <= MAX_TEXT_CHARS:
        raise ValueError(f"text must contain 1–{MAX_TEXT_CHARS} characters")
    result: dict[str, Any] = {
        "text": text,
        "instruction": _short({"instruction": str(body.get("instruction", "")).strip()},
                              "instruction", "", MAX_INSTRUCTION_CHARS,
                              f"instruction must contain at most {MAX_INSTRUCTION_CHARS} "
                              "characters"),
        "language": _short(body, "language", "Auto", MAX_LANGUAGE_CHARS, "language is too long"),
        "speaker": _short(body, "speaker", "", MAX_SPEAKER_CHARS, "speaker is too long"),
        "seed": _seed(body), "reference_audio": None, "reference_text": ""}
    result.update(_reference(body))
    return result


def _refuse_unknown(body: dict, *, seed: bool, reference: bool) -> None:
    allowed = {"model", "text", "instruction", "language", "speaker"}
    if seed:
        allowed.add("seed")
    if reference:
        allowed |= {"reference_audio", "reference_text"}
    unknown = set(body) - allowed
    if unknown:
        raise ValueError("Unknown speech fields for this engine: "
                         + ", ".join(sorted(unknown)))


def _short(body: dict, key: str, default: str, limit: int, message: str) -> str:
    """A text field of at most `limit` characters; `message` when longer."""
    value = str(body.get(key, default))
    if len(value) > limit:
        raise ValueError(message)
    return value


def _seed(body: dict) -> int | None:
    """The request's seed, or None when it names none."""
    value = body.get("seed")
    if value is None:
        return None
    if type(value) is not int or not 0 <= value <= MAX_SEED:
        raise ValueError(f"seed must be an integer from 0 to {MAX_SEED}")
    return value


def _reference(body: dict) -> dict[str, Any]:
    """The reference clip and its transcript, decoded and checked; empty when none."""
    if not body.get("reference_audio"):
        if body.get("reference_text"):
            raise ValueError("reference_text needs reference_audio")
        return {}
    try:
        audio = base64.b64decode(str(body["reference_audio"]), validate=True)
    except (binascii.Error, ValueError):
        raise ValueError("reference_audio must be base64-encoded audio") from None
    if not 0 < len(audio) <= MAX_REFERENCE_BYTES:
        raise ValueError("reference_audio must be at most 8 MB")
    if not audio.startswith(b"RIFF"):
        raise ValueError("reference_audio must be a WAV file")
    text = str(body.get("reference_text", "")).strip()
    if len(text) > MAX_TEXT_CHARS:
        raise ValueError(f"reference_text must contain at most {MAX_TEXT_CHARS} characters")
    return {"reference_audio": audio, "reference_text": text}


def wav_result(model: str, mode: str, samples, sample_rate: int) -> dict:
    """The speech contract's answer: `samples` as one 16-bit WAV, base64-encoded."""
    import soundfile as sf

    audio = io.BytesIO()
    sf.write(audio, samples, sample_rate, format="WAV", subtype="PCM_16")
    return {"model": model, "mode": mode, "sample_rate": sample_rate,
            "data": [{"mime_type": "audio/wav",
                      "b64_wav": base64.b64encode(audio.getvalue()).decode()}]}


def apply_seed(seed: int | None) -> None:
    """Fix the random draw for the next generation, on every device."""
    if seed is None:
        return
    import torch

    torch.manual_seed(seed)  # seeds CPU, CUDA and MPS generators alike


class ReferenceFile:
    """The reference clip as a private temporary WAV, removed afterwards.

    Runtimes that clone a voice want a path, not bytes. The file lives only
    for one generation and is readable by this account alone.
    """

    def __init__(self, audio: bytes | None) -> None:
        self.audio = audio
        self.path: str | None = None

    def __enter__(self) -> str | None:
        if self.audio is None:
            return None
        import os
        import tempfile

        handle, self.path = tempfile.mkstemp(prefix="ai-lab-ref-", suffix=".wav")
        with os.fdopen(handle, "wb") as out:
            out.write(self.audio)
        return self.path

    def __exit__(self, *_exc) -> None:
        if self.path:
            import os

            with contextlib.suppress(FileNotFoundError):
                os.unlink(self.path)
