"""MuLaCover generation from a bounded user-supplied WAV reference."""
from __future__ import annotations

import base64
import binascii
import secrets
import tempfile
import wave
from pathlib import Path
from threading import Lock

MAX_SOURCE_BYTES = 25 * 1024 * 1024


class MulaCoverBackend:
    def __init__(self, bundle_root: Path, checkpoint_path: Path,
                 checkpoint_subdir: str, output_root: Path,
                 topk: int, temperature: float, cfg_scale: float) -> None:
        import torch
        from mulacover import MuLaCoverGenPipeline

        if not bundle_root.is_dir() or not checkpoint_path.is_dir():
            raise ValueError("MuLaCover bundle or checkpoint is absent")
        if (Path(checkpoint_subdir).name != checkpoint_subdir or
                (bundle_root / checkpoint_subdir).resolve() != checkpoint_path.resolve()):
            raise ValueError("MuLaCover bundle does not point to the configured checkpoint")
        if not 1 <= topk <= 1000 or not 0 <= temperature <= 5 or not 0 < cfg_scale <= 10:
            raise ValueError("MuLaCover inference settings are invalid")
        output_root.mkdir(parents=True, exist_ok=True)
        self.output_root = output_root
        self.model_name = checkpoint_path.name
        self.topk = topk
        self.temperature = temperature
        self.cfg_scale = cfg_scale
        self.torch = torch
        device = torch.device("cuda:0")
        self.pipeline = MuLaCoverGenPipeline.from_pretrained(
            str(bundle_root), device=device,
            dtype={"mulacover": torch.bfloat16, "codec": torch.float32,
                   "qwen": torch.float32, "transcriptor": torch.float32},
            lazy_load=True)
        self.lock = Lock()

    def generate(self, body: dict) -> dict:
        request = self._validate(body)
        with tempfile.TemporaryDirectory(dir=self.output_root) as directory:
            root = Path(directory)
            reference = root / "reference.wav"
            lyrics = root / "lyrics.txt"
            tags = root / "tags.txt"
            output = root / "cover.wav"
            reference.write_bytes(request["audio"])
            lyrics.write_text(request["lyrics"] + "\n")
            tags.write_text(request["prompt"] + "\n")
            with self.lock, self.torch.no_grad():
                self.torch.manual_seed(request["seed"])
                self.torch.cuda.manual_seed_all(request["seed"])
                self.pipeline(
                    {"ref_audio": str(reference), "lyrics": str(lyrics),
                     "tags": str(tags)}, save_path=str(output),
                    topk=self.topk, temperature=self.temperature,
                    cfg_scale=self.cfg_scale)
            audio = output.read_bytes()
            if not audio.startswith(b"RIFF"):
                raise RuntimeError("MuLaCover did not return WAV audio")
            with wave.open(str(output)) as wav:
                duration = wav.getnframes() / wav.getframerate()
            return {"model": self.model_name, "seed": request["seed"],
                    "duration": duration,
                    "data": [{"mime_type": "audio/wav",
                              "b64_wav": base64.b64encode(audio).decode()}]}

    def _validate(self, body: dict) -> dict:
        allowed = {"model", "prompt", "lyrics", "instrumental", "seed",
                   "reference_audio_base64"}
        unknown = set(body) - allowed
        if unknown:
            raise ValueError("Unknown MuLaCover fields: " + ", ".join(sorted(unknown)))
        if body.get("model") not in (None, self.model_name):
            raise ValueError(f"this engine serves {self.model_name}")
        prompt = body.get("prompt", "")
        lyrics = body.get("lyrics", "")
        if not isinstance(prompt, str) or not 1 <= len(prompt.strip()) <= 4000:
            raise ValueError("style tags must contain 1–4000 characters")
        if not isinstance(lyrics, str) or not 1 <= len(lyrics.strip()) <= 12000:
            raise ValueError("lyrics must contain 1–12000 characters")
        if body.get("instrumental") is not False:
            raise ValueError("MuLaCover requires a vocal arrangement")
        encoded = body.get("reference_audio_base64", "")
        if not isinstance(encoded, str) or not encoded or len(encoded) > MAX_SOURCE_BYTES * 4 // 3 + 4:
            raise ValueError("reference WAV is missing or too large")
        try:
            audio = base64.b64decode(encoded, validate=True)
        except binascii.Error:
            raise ValueError("reference audio is not valid base64") from None
        if not audio.startswith(b"RIFF") or len(audio) > MAX_SOURCE_BYTES:
            raise ValueError("reference audio must be a WAV of at most 25 MiB")
        seed = body.get("seed", secrets.randbelow(2**32))
        if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
            raise ValueError("seed must be a 32-bit unsigned integer")
        return {"prompt": prompt.strip(), "lyrics": lyrics.strip(),
                "audio": audio, "seed": seed}
