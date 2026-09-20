#!/usr/bin/env python3
"""Isolated Qwen3-TTS server used through AI-Lab's leased gateway."""
from __future__ import annotations

import argparse
import base64
import io
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock

PATH = "/v1/audio/speech/generations"
MODES = {"voice-design", "custom-voice"}


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


class QwenTtsBackend:
    def __init__(self, model_path: Path, mode: str) -> None:
        import torch
        from qwen_tts import Qwen3TTSModel

        if mode not in MODES:
            raise ValueError(f"Unsupported Qwen3-TTS mode: {mode}")
        if not model_path.is_dir():
            raise ValueError("Qwen3-TTS model directory is absent")
        device = "cuda:0" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu")
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        self.model = Qwen3TTSModel.from_pretrained(
            str(model_path), device_map=device, dtype=dtype,
            attn_implementation="sdpa")
        self.mode = mode
        self.model_name = model_path.name
        self.lock = Lock()

    def generate(self, body: dict) -> dict:
        import soundfile as sf

        request = validate_payload(body)
        if body.get("model") not in (None, self.model_name):
            raise ValueError(f"this engine serves {self.model_name}")
        with self.lock:
            if self.mode == "voice-design":
                if not request["instruction"]:
                    raise ValueError("VoiceDesign requires a voice instruction")
                wavs, sample_rate = self.model.generate_voice_design(
                    text=request["text"], language=request["language"],
                    instruct=request["instruction"])
            else:
                speakers = self.model.get_supported_speakers()
                by_name = {name.casefold(): name for name in speakers}
                requested = request["speaker"] or (speakers[0] if speakers else "")
                speaker = by_name.get(requested.casefold())
                if speaker is None:
                    raise ValueError("speaker must be one of " + ", ".join(speakers))
                wavs, sample_rate = self.model.generate_custom_voice(
                    text=request["text"], language=request["language"],
                    speaker=speaker, instruct=request["instruction"])
        audio = io.BytesIO()
        sf.write(audio, wavs[0], sample_rate, format="WAV", subtype="PCM_16")
        return {"model": self.model_name, "mode": self.mode,
                "sample_rate": sample_rate, "data": [{
                    "mime_type": "audio/wav",
                    "b64_wav": base64.b64encode(audio.getvalue()).decode()}]}


class Handler(BaseHTTPRequestHandler):
    backend: QwenTtsBackend | None = None

    def do_GET(self):
        if self.path == "/health":
            self._json(200, {"status": "ok"})
        else:
            self._json(404, {"error": {"message": "not found"}})

    def do_POST(self):
        if self.path != PATH:
            self._json(404, {"error": {"message": "not found"}})
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 16384:
                raise ValueError("speech request is empty or too large")
            body = json.loads(self.rfile.read(size))
            if not isinstance(body, dict):
                raise ValueError("speech request must be an object")
            self._json(200, self.backend.generate(body))
        except (ValueError, TypeError) as error:
            self._json(400, {"error": {"message": str(error)}})
        except Exception as error:
            self._json(500, {"error": {"message": str(error)}})

    def _json(self, status: int, body: dict):
        payload = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--mode", choices=sorted(MODES), required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    Handler.backend = QwenTtsBackend(args.model_path, args.mode)
    ThreadingHTTPServer(("0.0.0.0", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
