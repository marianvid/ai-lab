"""Local Kokoro speech and voice files behind the common speech contract."""
from __future__ import annotations

import re
from pathlib import Path
from threading import Lock

from .contract import validate_payload, wav_result


class KokoroBackend:
    def __init__(self, checkpoint: Path, language_code: str,
                 default_voice: str, repo_id: str) -> None:
        import torch
        from kokoro import KModel, KPipeline

        if not checkpoint.is_file():
            raise ValueError("Kokoro checkpoint is absent")
        model_path = checkpoint.parent
        device = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu")
        self.device = device
        self.model = KModel(
            repo_id=repo_id, config=str(model_path / "config.json"),
            model=str(checkpoint)).to(device).eval()
        self.pipeline = KPipeline(
            lang_code=language_code, repo_id=repo_id,
            model=self.model, device=device)
        self.model_path = model_path
        self.model_name = model_path.name
        self.default_voice = default_voice
        self.torch = torch
        self.lock = Lock()

    def generate(self, body: dict) -> dict:
        request = validate_payload(body)
        if body.get("model") not in (None, self.model_name):
            raise ValueError(f"this engine serves {self.model_name}")
        if request["instruction"]:
            raise ValueError("Kokoro does not support voice instructions")
        if request["language"] not in ("Auto", "English"):
            raise ValueError("this Kokoro instance is configured for English")
        voice = request["speaker"] or self.default_voice
        if not re.fullmatch(r"[A-Za-z0-9_]+", voice):
            raise ValueError("voice must be a name from this checkpoint")
        voice_path = self.model_path / "voices" / f"{voice}.pt"
        if not voice_path.is_file():
            raise ValueError(f"Unknown voice: {voice}")
        with self.lock, self.torch.inference_mode():
            chunks = [item.audio for item in self.pipeline(
                request["text"], voice=str(voice_path), speed=1.0)]
        if not chunks:
            raise RuntimeError("Kokoro returned no audio")
        samples = self.torch.cat(chunks).detach().float().cpu().numpy()
        return wav_result(self.model_name, "kokoro", samples, 24000)
