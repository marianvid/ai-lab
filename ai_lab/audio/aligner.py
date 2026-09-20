"""Local transcript/audio alignment and a small JSON result contract."""
from __future__ import annotations

from pathlib import Path
from threading import Lock

SUPPORTED_LANGUAGES = frozenset({
    "Chinese", "English", "Cantonese", "French", "German", "Italian",
    "Japanese", "Korean", "Portuguese", "Russian", "Spanish"})


class QwenAlignBackend:
    owner = "qwen-align"
    path = "/v1/audio/alignments"

    def __init__(self, model_path: str, _precision: str = "bf16") -> None:
        import torch
        from qwen_asr import Qwen3ForcedAligner

        if not Path(model_path).is_dir():
            raise ValueError("Alignment checkpoint directory is absent")
        device = "cuda:0" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu")
        self.model = Qwen3ForcedAligner.from_pretrained(
            model_path, dtype=torch.bfloat16 if device != "cpu" else torch.float32,
            device_map=device)
        self.torch = torch
        self.lock = Lock()

    def align(self, audio_path: str, text: str, language: str) -> list[dict]:
        text = text.strip()
        if not 1 <= len(text) <= 10000:
            raise ValueError("transcript must contain 1–10000 characters")
        if language not in SUPPORTED_LANGUAGES:
            raise ValueError("unsupported alignment language: " + language)
        with self.lock, self.torch.inference_mode():
            results = self.model.align(audio=audio_path, text=text,
                                       language=language)
        return [{"text": item.text, "start": item.start_time,
                 "end": item.end_time} for item in results[0]]
