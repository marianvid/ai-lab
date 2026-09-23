"""Isolated YuE2 pipeline and its browser/agent music response."""
from __future__ import annotations

import atexit
import base64
import secrets
import tempfile
import wave
from pathlib import Path
from threading import Lock

from .yue2_web_backend import was_truncated


class Yue2Backend:
    def __init__(self, model_path: Path, vae_path: Path,
                 output_root: Path, cot: str, memory_budget_gib: float,
                 device: str = "cuda") -> None:
        from yue2 import YuE2Pipeline

        if not model_path.is_dir() or not vae_path.is_dir():
            raise ValueError("YuE2 model or VAE is absent")
        if cot not in {"full", "melody"} or not 4 <= memory_budget_gib <= 64:
            raise ValueError("YuE2 inference settings are invalid")
        output_root.mkdir(parents=True, exist_ok=True)
        self.output_root = output_root
        self.model_name = model_path.name
        self.cot = cot
        self.lock = Lock()
        self.context = YuE2Pipeline.from_pretrained(
            str(model_path), vae=str(vae_path), device=device,
            memory_budget_gib=memory_budget_gib, local_files_only=True)
        self.pipeline = self.context.__enter__()
        atexit.register(self.context.__exit__, None, None, None)

    def generate(self, body: dict) -> dict:
        request = self._validate(body)
        with tempfile.TemporaryDirectory(dir=self.output_root) as directory:
            root = Path(directory)
            args = {"style": request["prompt"], "lyrics": request["lyrics"],
                    "cot": self.cot, "seed": request["seed"], "id": "song"}
            if request["abc"]:
                args["abc"] = request["abc"]
            with self.lock:
                song = self.pipeline(**args)
                song.save_artifacts(root)
                import soundfile as sf
                samples, sample_rate = sf.read(root / "audio.flac", dtype="float32")
                output = root / "result.wav"
                sf.write(output, samples, sample_rate, subtype="PCM_16")
            audio = output.read_bytes()
            if not audio.startswith(b"RIFF"):
                raise RuntimeError("YuE2 did not return WAV audio")
            with wave.open(str(output)) as wav:
                duration = wav.getnframes() / wav.getframerate()
            score = root / "score.abc"
            abc = score.read_text() if score.is_file() else ""
            return {"model": self.model_name, "seed": request["seed"],
                    "duration": duration,
                    "truncated": was_truncated(song.truncated),
                    "score_abc": abc,
                    "data": [{"mime_type": "audio/wav",
                              "b64_wav": base64.b64encode(audio).decode()}]}

    def _validate(self, body: dict) -> dict:
        allowed = {"model", "prompt", "lyrics", "instrumental", "seed", "abc"}
        unknown = set(body) - allowed
        if unknown:
            raise ValueError("Unknown YuE2 fields: " + ", ".join(sorted(unknown)))
        if body.get("model") not in (None, self.model_name):
            raise ValueError(f"this engine serves {self.model_name}")
        prompt = body.get("prompt", "")
        lyrics = body.get("lyrics", "")
        abc = body.get("abc", "")
        if not isinstance(prompt, str) or not 1 <= len(prompt.strip()) <= 4000:
            raise ValueError("style must contain 1–4000 characters")
        if not isinstance(lyrics, str) or not 1 <= len(lyrics.strip()) <= 12000:
            raise ValueError("lyrics must contain 1–12000 characters")
        if not isinstance(abc, str) or len(abc) > 40000:
            raise ValueError("ABC score is too large")
        if body.get("instrumental") is not False:
            raise ValueError("YuE2 needs lyric-conditioned generation")
        seed = body.get("seed", secrets.randbelow(2**32))
        if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
            raise ValueError("seed must be a 32-bit unsigned integer")
        return {"prompt": prompt.strip(), "lyrics": lyrics.strip(),
                "abc": abc, "seed": seed}
