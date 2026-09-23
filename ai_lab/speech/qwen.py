"""Qwen3-TTS checkpoint modes behind the common speech contract."""
from __future__ import annotations

from pathlib import Path
from threading import Lock

from .contract import apply_seed, validate_payload, wav_result

MODES = {"voice-design", "custom-voice"}


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
        request = validate_payload(body, seed=True)
        if body.get("model") not in (None, self.model_name):
            raise ValueError(f"this engine serves {self.model_name}")
        with self.lock:
            apply_seed(request["seed"])
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
        return wav_result(self.model_name, self.mode, wavs[0], sample_rate)
