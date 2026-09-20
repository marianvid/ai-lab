"""HeartMuLa pipeline with a bounded request and WAV result contract."""
from __future__ import annotations

import base64
import secrets
import tempfile
import wave
from pathlib import Path
from threading import Lock


class HeartMulaBackend:
    def __init__(self, bundle_root: Path, checkpoint_path: Path,
                 checkpoint_subdir: str, output_root: Path, version: str,
                 topk: int, temperature: float, cfg_scale: float) -> None:
        import torch
        from heartlib import HeartMuLaGenPipeline

        if not bundle_root.is_dir() or not checkpoint_path.is_dir():
            raise ValueError("HeartMuLa bundle or checkpoint is absent")
        if (Path(checkpoint_subdir).name != checkpoint_subdir or
                (bundle_root / checkpoint_subdir).resolve() != checkpoint_path.resolve()):
            raise ValueError("HeartMuLa bundle does not point to the configured checkpoint")
        if not version or not 1 <= topk <= 1000 or not 0 <= temperature <= 5:
            raise ValueError("HeartMuLa inference settings are invalid")
        if not 0 < cfg_scale <= 10:
            raise ValueError("HeartMuLa guidance scale is invalid")
        output_root.mkdir(parents=True, exist_ok=True)
        self.output_root = output_root
        self.model_name = checkpoint_path.name
        self.topk = topk
        self.temperature = temperature
        self.cfg_scale = cfg_scale
        self.torch = torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pipeline = HeartMuLaGenPipeline.from_pretrained(
            str(bundle_root), device={"mula": device, "codec": device},
            dtype={"mula": torch.bfloat16 if device.type == "cuda"
                   else torch.float32, "codec": torch.float32},
            version=version, lazy_load=True)
        self.lock = Lock()

    def generate(self, body: dict) -> dict:
        request = self._validate(body)
        with tempfile.TemporaryDirectory(dir=self.output_root) as directory:
            root = Path(directory)
            lyrics_path = root / "lyrics.txt"
            tags_path = root / "tags.txt"
            output = root / "result.wav"
            lyrics_path.write_text(request["lyrics"] + "\n")
            tags_path.write_text(request["prompt"] + "\n")
            with self.lock, self.torch.no_grad():
                self.torch.manual_seed(request["seed"])
                if self.torch.cuda.is_available():
                    self.torch.cuda.manual_seed_all(request["seed"])
                self.pipeline(
                    {"lyrics": str(lyrics_path), "tags": str(tags_path)},
                    max_audio_length_ms=round(request["duration"] * 1000),
                    save_path=str(output), topk=self.topk,
                    temperature=self.temperature, cfg_scale=self.cfg_scale)
            audio = output.read_bytes()
            if not audio.startswith(b"RIFF"):
                raise RuntimeError("HeartMuLa did not return WAV audio")
            with wave.open(str(output)) as wav:
                actual_duration = wav.getnframes() / wav.getframerate()
        return {"model": self.model_name, "seed": request["seed"],
                "duration": actual_duration,
                "data": [{"mime_type": "audio/wav",
                          "b64_wav": base64.b64encode(audio).decode()}]}

    def _validate(self, body: dict) -> dict:
        allowed = {"model", "prompt", "lyrics", "instrumental", "duration", "seed"}
        unknown = set(body) - allowed
        if unknown:
            raise ValueError("Unknown HeartMuLa fields: " + ", ".join(sorted(unknown)))
        if body.get("model") not in (None, self.model_name):
            raise ValueError(f"this engine serves {self.model_name}")
        prompt = str(body.get("prompt", "")).strip()
        lyrics = str(body.get("lyrics", "")).strip()
        if not 1 <= len(prompt) <= 4000 or not 1 <= len(lyrics) <= 12000:
            raise ValueError("HeartMuLa needs style tags and lyrics")
        if body.get("instrumental") is not False:
            raise ValueError("HeartMuLa needs lyric-conditioned generation")
        duration = body.get("duration", 60)
        if type(duration) not in (int, float) or not 5 <= duration <= 180:
            raise ValueError("duration must be between 5 and 180 seconds")
        seed = body.get("seed", secrets.randbelow(2**32))
        if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
            raise ValueError("seed must be a 32-bit unsigned integer")
        return {"prompt": prompt, "lyrics": lyrics,
                "duration": duration, "seed": seed}
