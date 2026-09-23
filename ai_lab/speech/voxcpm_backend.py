"""Local VoxCPM generation behind the shared speech response contract."""
from __future__ import annotations

from pathlib import Path
from threading import Lock

from .contract import ReferenceFile, apply_seed, validate_payload, wav_result


class VoxCpmBackend:
    def __init__(self, model_path: Path, cfg_value: float,
                 inference_timesteps: int) -> None:
        from voxcpm import VoxCPM

        if not model_path.is_dir():
            raise ValueError("VoxCPM checkpoint directory is absent")
        if not 0 < cfg_value <= 10 or not 1 <= inference_timesteps <= 100:
            raise ValueError("VoxCPM inference settings are out of range")
        self.model = VoxCPM.from_pretrained(
            str(model_path), local_files_only=True,
            load_denoiser=False, optimize=False)
        self.model_name = model_path.name
        self.cfg_value = cfg_value
        self.inference_timesteps = inference_timesteps
        self.lock = Lock()

    def generate(self, body: dict) -> dict:
        request = validate_payload(body, seed=True, reference=True)
        if body.get("model") not in (None, self.model_name):
            raise ValueError(f"this engine serves {self.model_name}")
        if request["speaker"]:
            raise ValueError("VoxCPM has no named voices; clone one with reference audio")
        instruction = request["instruction"]
        text = f"({instruction}){request['text']}" if instruction else request["text"]
        with self.lock, ReferenceFile(request["reference_audio"]) as reference:
            clone = {}
            if reference:
                # The reference alone clones the voice; its transcript, when
                # given, also lets VoxCPM continue in the same delivery.
                clone["reference_wav_path"] = reference
                if request["reference_text"]:
                    clone["prompt_wav_path"] = reference
                    clone["prompt_text"] = request["reference_text"]
            apply_seed(request["seed"])
            samples = self.model.generate(
                text=text, cfg_value=self.cfg_value,
                inference_timesteps=self.inference_timesteps,
                normalize=False, denoise=False, retry_badcase=True,
                retry_badcase_max_times=1, **clone)
        return wav_result(self.model_name, "voxcpm", samples,
                          self.model.tts_model.sample_rate)
