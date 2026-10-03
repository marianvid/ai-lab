"""Stable Audio 3 Medium: instrumental music from a text prompt.

The model is loaded from AI-Lab's own model folder, never from the Hugging
Face cache: the text encoder sits in a subfolder of the same download, and
the configuration is pointed at it before loading.
"""
from __future__ import annotations

import base64
import json
import secrets
from io import BytesIO
from pathlib import Path
from threading import Lock

ALLOWED_FIELDS = {"model", "prompt", "negative_prompt", "duration", "seed",
                  "instrumental", "lyrics"}


class StableAudio3Backend:
    def __init__(self, model_dir: Path, steps: int, cfg_scale: float) -> None:
        import torch
        from stable_audio_3.loading_utils import load_diffusion_cond
        from stable_audio_3.model import StableAudioModel

        config_path = model_dir / "model_config.json"
        checkpoint = model_dir / "model.safetensors"
        if not config_path.is_file() or not checkpoint.is_file():
            raise ValueError("Stable Audio 3 files are absent")
        if not 1 <= steps <= 100 or not 0 < cfg_scale <= 10:
            raise ValueError("Stable Audio 3 settings are invalid")
        config = local_config(json.loads(config_path.read_text()), model_dir)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        half = device == "cuda"
        model = load_diffusion_cond(config, str(checkpoint), device=device,
                                    model_half=half)
        model.use_lora = False
        model.lora_names = []
        self.model = StableAudioModel(model, config, device, half)
        self.sample_rate = int(config["sample_rate"])
        # The longest audio the model is configured for (380 s for Medium).
        # generate() defaults to 120 s and silently cuts anything longer, so
        # the configured length is passed explicitly, as the upstream CLI does.
        self.sample_size = int(config["sample_size"])
        self.model_name = model_dir.name
        self.steps = steps
        self.cfg_scale = cfg_scale
        self.lock = Lock()

    def generate(self, body: dict) -> dict:
        request = validate(body, self.model_name)
        with self.lock:
            audio = self.model.generate(
                prompt=request["prompt"],
                negative_prompt=request["negative_prompt"] or None,
                duration=request["duration"], steps=self.steps,
                cfg_scale=self.cfg_scale, seed=request["seed"],
                sample_size=self.sample_size)
        wav, seconds = to_wav(audio, self.sample_rate)
        return {"model": self.model_name, "seed": request["seed"],
                "duration": seconds, "instrumental": True,
                "data": [{"mime_type": "audio/wav",
                          "b64_wav": base64.b64encode(wav).decode()}]}


def local_config(config: dict, model_dir: Path) -> dict:
    """Point every Hugging Face reference in the conditioners at `model_dir`."""
    for item in config.get("model", {}).get("conditioning", {}).get("configs", []):
        settings = item.get("config", {})
        if "repo_id" in settings:
            settings["model_path"] = str(model_dir)
            settings.pop("repo_id")
    return config


def validate(body: dict, model_name: str) -> dict:
    unknown = set(body) - ALLOWED_FIELDS
    if unknown:
        raise ValueError("Unknown Stable Audio 3 fields: " + ", ".join(sorted(unknown)))
    if body.get("model") not in (None, model_name):
        raise ValueError(f"this engine serves {model_name}")
    prompt = body.get("prompt", "")
    if not isinstance(prompt, str) or not 1 <= len(prompt.strip()) <= 4000:
        raise ValueError("prompt must contain 1–4000 characters")
    negative = body.get("negative_prompt", "")
    if not isinstance(negative, str) or len(negative) > 4000:
        raise ValueError("negative_prompt must be text of at most 4000 characters")
    if body.get("instrumental", True) is not True:
        raise ValueError("Stable Audio 3 makes instrumentals only; send instrumental: true")
    if str(body.get("lyrics", "")).strip():
        raise ValueError("Stable Audio 3 does not sing; leave lyrics out")
    duration = body.get("duration", 30)
    if type(duration) not in (int, float) or not 1 <= duration <= 380:
        raise ValueError("duration must be between 1 and 380 seconds")
    seed = body.get("seed", secrets.randbelow(2**31))
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed must be a 32-bit unsigned integer")
    return {"prompt": prompt.strip(), "negative_prompt": negative.strip(),
            "duration": duration, "seed": seed}


def to_wav(audio, sample_rate: int) -> tuple[bytes, float]:
    """Model output [batch, channels, samples] → 16-bit stereo WAV bytes."""
    import soundfile as sf
    samples = audio[0].float().clamp(-1, 1).cpu().numpy().T
    out = BytesIO()
    sf.write(out, samples, sample_rate, format="WAV", subtype="PCM_16")
    return out.getvalue(), samples.shape[0] / sample_rate
