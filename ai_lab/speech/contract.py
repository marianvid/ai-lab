"""Request validation and WAV response shared by speech engines."""
from __future__ import annotations

import base64
import io

PATH = "/v1/audio/speech/generations"


def validate_payload(body: dict) -> dict:
    allowed = {"model", "text", "instruction", "language", "speaker"}
    unknown = set(body) - allowed
    if unknown:
        raise ValueError("Unknown speech fields: " + ", ".join(sorted(unknown)))
    text = str(body.get("text", "")).strip()
    if not 1 <= len(text) <= 2000:
        raise ValueError("text must contain 1–2000 characters")
    instruction = str(body.get("instruction", "")).strip()
    if len(instruction) > 1000:
        raise ValueError("instruction must contain at most 1000 characters")
    language = str(body.get("language", "Auto"))
    if len(language) > 32:
        raise ValueError("language is too long")
    speaker = str(body.get("speaker", ""))
    if len(speaker) > 80:
        raise ValueError("speaker is too long")
    return {"text": text, "instruction": instruction,
            "language": language, "speaker": speaker}


def wav_result(model: str, mode: str, samples, sample_rate: int) -> dict:
    import soundfile as sf

    audio = io.BytesIO()
    sf.write(audio, samples, sample_rate, format="WAV", subtype="PCM_16")
    return {"model": model, "mode": mode, "sample_rate": sample_rate,
            "data": [{"mime_type": "audio/wav",
                      "b64_wav": base64.b64encode(audio.getvalue()).decode()}]}
