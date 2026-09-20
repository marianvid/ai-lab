"""Supervised Khala CLI invocation with the AI-Lab music result contract."""
from __future__ import annotations

import base64
import os
import secrets
import subprocess
import tempfile
import wave
from pathlib import Path
from threading import Lock


class KhalaBackend:
    def __init__(self, generator_script: Path, model_path: Path,
                 output_root: Path, default_bucket: int,
                 maximum_bucket: int) -> None:
        if not generator_script.is_file() or not model_path.is_dir():
            raise ValueError("Khala generator or checkpoint is absent")
        if not 0 <= default_bucket <= maximum_bucket <= 20:
            raise ValueError("Khala length buckets are invalid")
        output_root.mkdir(parents=True, exist_ok=True)
        self.script = generator_script
        self.model_path = model_path
        self.output_root = output_root
        self.model_name = model_path.name
        self.default_bucket = default_bucket
        self.maximum_bucket = maximum_bucket
        self.lock = Lock()

    def generate(self, body: dict) -> dict:
        request = self._validate(body)
        with tempfile.TemporaryDirectory(dir=self.output_root) as directory:
            output = Path(directory) / "result.wav"
            command = [
                os.sys.executable, "-u", str(self.script),
                "--language", "Instrumental" if request["instrumental"]
                else request["vocal_language"],
                "--description", request["prompt"],
                "--duration", str(request["length_bucket"]),
                "--seed", str(request["seed"]), "--out", str(output)]
            if not request["instrumental"]:
                command += ["--lyrics", request["lyrics"]]
            environment = os.environ.copy()
            environment.update({"KHALA_BACKEND": "vanilla",
                                "KHALA_VANILLA_WEIGHTS": str(self.model_path)})
            with self.lock:
                completed = subprocess.run(
                    command, cwd=self.script.parent.parent, env=environment,
                    capture_output=True, text=True, timeout=1200)
            if completed.returncode:
                raise RuntimeError("Khala generation failed: " +
                                   completed.stderr[-700:])
            audio = output.read_bytes()
            if not audio.startswith(b"RIFF"):
                raise RuntimeError("Khala did not return WAV audio")
            with wave.open(str(output)) as wav:
                duration = wav.getnframes() / wav.getframerate()
        return {"model": self.model_name, "seed": request["seed"],
                "duration": duration, "length_bucket": request["length_bucket"],
                "data": [{"mime_type": "audio/wav",
                          "b64_wav": base64.b64encode(audio).decode()}]}

    def _validate(self, body: dict) -> dict:
        allowed = {"model", "prompt", "lyrics", "instrumental", "seed",
                   "length_bucket", "vocal_language"}
        unknown = set(body) - allowed
        if unknown:
            raise ValueError("Unknown Khala fields: " + ", ".join(sorted(unknown)))
        if body.get("model") not in (None, self.model_name):
            raise ValueError(f"this engine serves {self.model_name}")
        prompt = str(body.get("prompt", "")).strip()
        lyrics = str(body.get("lyrics", "")).strip()
        if not 1 <= len(prompt) <= 4000 or len(lyrics) > 12000:
            raise ValueError("Khala prompt or lyrics length is invalid")
        instrumental = body.get("instrumental", not bool(lyrics))
        if not isinstance(instrumental, bool):
            raise ValueError("instrumental must be true or false")
        language = str(body.get("vocal_language", "English"))
        if language not in {"Chinese", "English", "Japanese", "Korean",
                            "Cantonese"}:
            raise ValueError("Unsupported Khala vocal language")
        bucket = body.get("length_bucket", self.default_bucket)
        if type(bucket) is not int or not 0 <= bucket <= self.maximum_bucket:
            raise ValueError("Khala length bucket is out of range")
        seed = body.get("seed", secrets.randbelow(2**32))
        if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
            raise ValueError("seed must be a 32-bit unsigned integer")
        return {"prompt": prompt, "lyrics": lyrics,
                "instrumental": instrumental, "vocal_language": language,
                "length_bucket": bucket, "seed": seed}
