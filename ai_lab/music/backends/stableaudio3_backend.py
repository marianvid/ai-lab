"""Stable Audio 3 Medium: instrumental music from a text prompt.

The model is loaded from AI-Lab's own model folder, never from the Hugging
Face cache: the text encoder sits in a subfolder of the same download, and
the configuration is pointed at it before loading.
"""
from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from threading import Lock

from ..request_fields import (
    MAX_PROMPT_CHARS,
    TextRule,
    WavRule,
    check_fields,
    decoded_wav,
    seconds,
    seed,
    text,
    wav_answer,
)

ALLOWED_FIELDS = {"model", "prompt", "negative_prompt", "duration", "seed",
                  "instrumental", "lyrics", "init_audio", "init_noise_level",
                  "inpaint_audio", "inpaint_mask_start_seconds",
                  "inpaint_mask_end_seconds"}
# One source recording, as a WAV: 25 MiB is about 2.5 minutes of 16-bit
# stereo at 44.1 kHz. Both sources together must also fit the media job
# input limit (`media.max_input_bytes`, 40 MiB by default).
MAX_SOURCE_BYTES = 25 * 1024 * 1024
# The longest song the Medium model is configured for, and the default.
DEFAULT_S, SHORTEST_S, LONGEST_S = 30, 1, 380
# The render settings the model is run with.
MAX_STEPS = 100
MAX_GUIDANCE = 10.0
# A seed drawn when none is sent stays below 2**31, as the library's own does.
DRAWN_SEEDS = 2**31
PROMPT = TextRule("prompt must contain 1–4000 characters", MAX_PROMPT_CHARS, shortest=1)
NEGATIVE = TextRule("negative_prompt must be text of at most 4000 characters", MAX_PROMPT_CHARS)


class StableAudio3Backend:
    """Stable Audio 3 Medium, loaded once, for instrumentals, restyles and inpainting."""

    def __init__(self, model_dir: Path, steps: int, cfg_scale: float) -> None:
        import torch
        from stable_audio_3.loading_utils import load_diffusion_cond
        from stable_audio_3.model import StableAudioModel

        config_path = model_dir / "model_config.json"
        checkpoint = model_dir / "model.safetensors"
        if not config_path.is_file() or not checkpoint.is_file():
            raise ValueError("Stable Audio 3 files are absent")
        if not 1 <= steps <= MAX_STEPS or not 0 < cfg_scale <= MAX_GUIDANCE:
            raise ValueError("Stable Audio 3 settings are invalid")
        config = local_config(json.loads(config_path.read_text()), model_dir)
        # The NVIDIA card on Linux, Metal on a Mac. Half precision only on
        # CUDA: Metal runs the model in full precision, as the library does.
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
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
        """One instrumental for the request, as base64 WAV."""
        request = validate(body, self.model_name)
        extra = {}
        if request["init_audio"] is not None:
            extra["init_audio"] = read_wav(request["init_audio"])
            extra["init_noise_level"] = request["init_noise_level"]
        if request["inpaint_audio"] is not None:
            extra["inpaint_audio"] = read_wav(request["inpaint_audio"])
            extra["inpaint_mask_start_seconds"] = request["inpaint_starts"]
            extra["inpaint_mask_end_seconds"] = request["inpaint_ends"]
        with self.lock:
            audio = self.model.generate(
                prompt=request["prompt"],
                negative_prompt=request["negative_prompt"] or None,
                duration=request["duration"], steps=self.steps,
                cfg_scale=self.cfg_scale, seed=request["seed"],
                sample_size=self.sample_size, **extra)
        wav, length = to_wav(audio, self.sample_rate)
        return wav_answer(wav, model=self.model_name, seed=request["seed"],
                          duration=length, instrumental=True)


def local_config(config: dict, model_dir: Path) -> dict:
    """Point every Hugging Face reference in the conditioners at `model_dir`."""
    for item in config.get("model", {}).get("conditioning", {}).get("configs", []):
        settings = item.get("config", {})
        if "repo_id" in settings:
            settings["model_path"] = str(model_dir)
            settings.pop("repo_id")
    return config


def validate(body: dict, model_name: str) -> dict:
    """Check a request and return it in the form `generate` uses."""
    _refuse_unknown_fields(body, model_name)
    duration = seconds(body, DEFAULT_S, SHORTEST_S, LONGEST_S)
    init_audio, noise = _restyle_source(body)
    inpaint_audio, starts, ends = _inpaint_source(body, duration)
    return {"prompt": text(body, "prompt", PROMPT),
            "negative_prompt": text(body, "negative_prompt", NEGATIVE),
            "duration": duration, "seed": seed(body, DRAWN_SEEDS),
            "init_audio": init_audio, "init_noise_level": noise,
            "inpaint_audio": inpaint_audio, "inpaint_starts": starts,
            "inpaint_ends": ends}


def _refuse_unknown_fields(body: dict, model_name: str) -> None:
    check_fields(body, ALLOWED_FIELDS, "Stable Audio 3", model_name)
    if body.get("instrumental", True) is not True:
        raise ValueError("Stable Audio 3 makes instrumentals only; send instrumental: true")
    if str(body.get("lyrics", "")).strip():
        raise ValueError("Stable Audio 3 does not sing; leave lyrics out")


def _restyle_source(body: dict) -> tuple[bytes | None, float | None]:
    """`init_audio` and how far the result may move from it (0–1, default 1)."""
    audio = source_wav(body, "init_audio")
    noise = body.get("init_noise_level")
    if noise is not None and audio is None:
        raise ValueError("init_noise_level needs init_audio")
    if audio is None:
        return None, noise
    noise = 1.0 if noise is None else noise
    if type(noise) not in (int, float) or not 0 <= noise <= 1:
        raise ValueError("init_noise_level must be a number from 0 to 1")
    return audio, noise


def _inpaint_source(body: dict, duration: float) -> tuple[bytes | None, list | None, list | None]:
    """`inpaint_audio` and the regions of it to regenerate."""
    audio = source_wav(body, "inpaint_audio")
    starts = body.get("inpaint_mask_start_seconds")
    ends = body.get("inpaint_mask_end_seconds")
    if audio is None:
        if starts is not None or ends is not None:
            raise ValueError("inpaint_mask_*_seconds need inpaint_audio")
        return None, starts, ends
    starts, ends = mask_regions(starts, ends, duration)
    return audio, starts, ends


def source_wav(body: dict, field: str) -> bytes | None:
    """A base64 WAV from the request, checked but not yet decoded to samples."""
    encoded = body.get(field)
    if encoded is None:
        return None
    return decoded_wav(encoded, WavRule(
        MAX_SOURCE_BYTES, f"{field} must be a base64 WAV of at most 25 MiB",
        f"{field} is not valid base64", f"{field} must be a WAV of at most 25 MiB"))


def mask_regions(starts, ends, duration: float) -> tuple[list, list]:
    """Regions to regenerate: one number each, or two lists of equal length."""
    if starts is None or ends is None:
        raise ValueError("inpaint_audio needs inpaint_mask_start_seconds and "
                         "inpaint_mask_end_seconds")
    starts, ends = _as_list(starts), _as_list(ends)
    if not starts or len(starts) != len(ends):
        raise ValueError("inpaint mask starts and ends must have the same length")
    if not all(_region_fits(start, end, duration)
               for start, end in zip(starts, ends, strict=True)):
        raise ValueError("each inpaint region needs 0 <= start < end <= duration")
    return [float(x) for x in starts], [float(x) for x in ends]


def _as_list(value) -> list:
    """One number, or a list of them, as a list."""
    return value if isinstance(value, list) else [value]


def _region_fits(start, end, duration: float) -> bool:
    """Both ends are numbers, in order, inside the song."""
    numbers = type(start) in (int, float) and type(end) in (int, float)
    return numbers and 0 <= start < end <= duration


def read_wav(data: bytes):
    """WAV bytes → (sample_rate, tensor [channels, samples]), as generate() wants."""
    import soundfile as sf
    import torch
    samples, rate = sf.read(BytesIO(data), dtype="float32", always_2d=True)
    return rate, torch.from_numpy(samples.T.copy())


def to_wav(audio, sample_rate: int) -> tuple[bytes, float]:
    """Model output [batch, channels, samples] → 16-bit stereo WAV bytes."""
    import soundfile as sf
    samples = audio[0].float().clamp(-1, 1).cpu().numpy().T
    out = BytesIO()
    sf.write(out, samples, sample_rate, format="WAV", subtype="PCM_16")
    return out.getvalue(), samples.shape[0] / sample_rate
