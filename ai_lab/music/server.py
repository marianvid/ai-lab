#!/usr/bin/env python3
"""Small, synchronous music API over an isolated ACE-Step environment.

The gateway holds its model lease for the entire generation. This prevents a
second model from evicting ACE-Step while it is writing audio. The upstream
runtime stays in its own environment and can be replaced independently.
"""
from __future__ import annotations

import argparse
import base64
import json
import secrets
import shutil
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock

PATH = "/v1/audio/music/generations"


class AceStepBackend:
    def __init__(self, project_root: Path, config_name: str,
                 model_path: Path, output_root: Path) -> None:
        from acestep.handler import AceStepHandler
        from acestep.llm_inference import LLMHandler

        checkpoint = (project_root / "checkpoints" / config_name).resolve()
        if checkpoint != model_path.resolve():
            raise ValueError("ACE-Step checkpoint does not match the configured model")
        self.output_root = output_root
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.model_name = model_path.name
        self.dit = AceStepHandler()
        status, ready = self.dit.initialize_service(
            project_root=str(project_root), config_path=config_name,
            device="auto")
        if not ready:
            raise RuntimeError(status)
        self.llm = LLMHandler()
        self.lock = Lock()

    def generate(self, body: dict) -> dict:
        from acestep.inference import GenerationConfig, GenerationParams, generate_music

        request = validate_payload(body)
        if body.get("model") not in (None, self.model_name):
            raise ValueError(f"this engine serves {self.model_name}")
        prompt = request["prompt"]
        lyrics = request["lyrics"]
        duration = request["duration"]
        seed = request["seed"]
        instrumental = request["instrumental"]
        language = request["vocal_language"]
        params = GenerationParams(
            caption=prompt, lyrics=lyrics, instrumental=instrumental,
            vocal_language=language, duration=duration, inference_steps=8,
            seed=seed, shift=3.0, thinking=False,
            use_cot_metas=False, use_cot_caption=False,
            use_cot_language=False)
        settings = GenerationConfig(batch_size=1, use_random_seed=False,
                                    seeds=[seed], audio_format="wav")
        output = self.output_root / uuid.uuid4().hex
        output.mkdir()
        try:
            with self.lock:
                result = generate_music(self.dit, self.llm, params, settings,
                                        save_dir=str(output))
            if not result.success or not result.audios:
                raise RuntimeError(result.error or "ACE-Step returned no audio")
            audio_path = Path(result.audios[0]["path"]).resolve()
            if not audio_path.is_relative_to(output.resolve()):
                raise RuntimeError("ACE-Step output escaped the job directory")
            audio = audio_path.read_bytes()
            if not audio.startswith(b"RIFF"):
                raise RuntimeError("ACE-Step did not return WAV audio")
            return {"model": self.model_name, "data": [{
                "mime_type": "audio/wav", "b64_wav": base64.b64encode(audio).decode()
            }], "seed": seed, "duration": duration}
        finally:
            shutil.rmtree(output)



def validate_payload(body: dict) -> dict:
    allowed = {"model", "prompt", "lyrics", "duration", "seed",
               "instrumental", "vocal_language"}
    unknown = set(body) - allowed
    if unknown:
        raise ValueError("Unknown music fields: " + ", ".join(sorted(unknown)))
    prompt = str(body.get("prompt", "")).strip()
    if not prompt or len(prompt) > 4000:
        raise ValueError("prompt must contain 1–4000 characters")
    lyrics = str(body.get("lyrics", "[Instrumental]"))
    if len(lyrics) > 12000:
        raise ValueError("lyrics must contain at most 12000 characters")
    duration = float(body.get("duration", 30))
    if not 5 <= duration <= 180:
        raise ValueError("duration must be between 5 and 180 seconds")
    seed = int(body["seed"] if "seed" in body else secrets.randbelow(2**32))
    if not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed must be a 32-bit unsigned integer")
    instrumental = body.get("instrumental", False)
    if not isinstance(instrumental, bool):
        raise ValueError("instrumental must be true or false")
    language = str(body.get("vocal_language", "unknown"))
    if len(language) > 32:
        raise ValueError("vocal_language is too long")
    return {"prompt": prompt, "lyrics": lyrics, "duration": duration,
            "seed": seed, "instrumental": instrumental,
            "vocal_language": language}


class Handler(BaseHTTPRequestHandler):
    backend: AceStepBackend | None = None

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
            if not 0 < size <= 65536:
                raise ValueError("music request is empty or too large")
            body = json.loads(self.rfile.read(size))
            if not isinstance(body, dict):
                raise ValueError("music request must be an object")
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
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    Handler.backend = AceStepBackend(args.project_root, args.config_name,
                                     args.model_path, args.output_root)
    ThreadingHTTPServer(("0.0.0.0", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
