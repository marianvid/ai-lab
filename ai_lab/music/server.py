#!/usr/bin/env python3
"""Small, synchronous music API over an isolated ACE-Step environment.

The gateway holds its model lease for the entire generation. This prevents a
second model from evicting ACE-Step while it is writing audio. The upstream
runtime stays in its own environment and can be replaced independently.
"""
from __future__ import annotations

import base64
import secrets
import shutil
import sys
import uuid
from pathlib import Path
from threading import Lock

# This file is launched directly by the runtime's own Python environment.
# Give it access to the small shared modules beside this script.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ai_lab.media.http_host import MUSIC_PATH, serve
from ai_lab.music.hosting import PORT, UI_PORT, parse
from ai_lab.music.request_fields import MAX_LYRICS_CHARS, MAX_PROMPT_CHARS, MAX_SEED
from ai_lab.network import ALL_INTERFACES

PATH = MUSIC_PATH
# What one request may ask for, beside the limits every music engine shares.
SHORTEST_S, LONGEST_S = 5, 180
MAX_LANGUAGE_CHARS = 32


class AceStepBackend:
    """ACE-Step 1.5: its music model (DiT) and language model, loaded once."""

    def __init__(self, project_root: Path, config_name: str,
                 model_path: Path, output_root: Path) -> None:
        from acestep.handler import AceStepHandler
        from acestep.llm_inference import LLMHandler

        checkpoint = (project_root / "checkpoints" / config_name).resolve()
        if checkpoint != model_path.resolve():
            raise ValueError("ACE-Step checkpoint does not match the configured model")
        self.output_root = output_root
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.model_name = model_path.name
        self.dit = AceStepHandler()
        status, ready = self.dit.initialize_service(
            project_root=str(project_root), config_path=config_name,
            device="auto")
        if not ready:
            raise RuntimeError(status)
        self.llm = LLMHandler()
        self.lock = Lock()

    def generate(self, body: dict) -> dict:
        """One piece of music for the request, as base64 WAV."""
        from acestep.inference import GenerationConfig, GenerationParams, generate_music

        request = validate_payload(body)
        if body.get("model") not in (None, self.model_name):
            raise ValueError(f"this engine serves {self.model_name}")
        prompt = request["prompt"]
        lyrics = request["lyrics"]
        duration = request["duration"]
        seed = request["seed"]
        instrumental = request["instrumental"]
        language = request["vocal_language"]
        params = GenerationParams(
            caption=prompt, lyrics=lyrics, instrumental=instrumental,
            vocal_language=language, duration=duration, inference_steps=8,
            seed=seed, shift=3.0, thinking=False,
            use_cot_metas=False, use_cot_caption=False,
            use_cot_language=False)
        settings = GenerationConfig(batch_size=1, use_random_seed=False,
                                    seeds=[seed], audio_format="wav")
        output = self.output_root / uuid.uuid4().hex
        output.mkdir()
        try:
            with self.lock:
                result = generate_music(self.dit, self.llm, params, settings,
                                        save_dir=str(output))
            if not result.success or not result.audios:
                raise RuntimeError(result.error or "ACE-Step returned no audio")
            audio_path = Path(result.audios[0]["path"]).resolve()
            if not audio_path.is_relative_to(output.resolve()):
                raise RuntimeError("ACE-Step output escaped the job directory")
            audio = audio_path.read_bytes()
            if not audio.startswith(b"RIFF"):
                raise RuntimeError("ACE-Step did not return WAV audio")
            return {"model": self.model_name, "data": [{
                "mime_type": "audio/wav", "b64_wav": base64.b64encode(audio).decode()
            }], "seed": seed, "duration": duration}
        finally:
            shutil.rmtree(output)



def validate_payload(body: dict) -> dict:
    """The request's fields, checked and with their defaults; ValueError says what is wrong."""
    allowed = {"model", "prompt", "lyrics", "duration", "seed",
               "instrumental", "vocal_language"}
    unknown = set(body) - allowed
    if unknown:
        raise ValueError("Unknown music fields: " + ", ".join(sorted(unknown)))
    prompt = str(body.get("prompt", "")).strip()
    if not prompt or len(prompt) > MAX_PROMPT_CHARS:
        raise ValueError("prompt must contain 1–4000 characters")
    lyrics = str(body.get("lyrics", "[Instrumental]"))
    if len(lyrics) > MAX_LYRICS_CHARS:
        raise ValueError("lyrics must contain at most 12000 characters")
    duration = float(body.get("duration", 30))
    if not SHORTEST_S <= duration <= LONGEST_S:
        raise ValueError("duration must be between 5 and 180 seconds")
    seed = int(body["seed"] if "seed" in body else secrets.randbelow(MAX_SEED + 1))
    if not 0 <= seed <= MAX_SEED:
        raise ValueError("seed must be a 32-bit unsigned integer")
    instrumental = body.get("instrumental", False)
    if not isinstance(instrumental, bool):
        raise ValueError("instrumental must be true or false")
    language = str(body.get("vocal_language", "unknown"))
    if len(language) > MAX_LANGUAGE_CHARS:
        raise ValueError("vocal_language is too long")
    return {"prompt": prompt, "lyrics": lyrics, "duration": duration,
            "seed": seed, "instrumental": instrumental,
            "vocal_language": language}


def main() -> None:
    """Load ACE-Step and its own editor, then answer music requests."""
    args = parse(("--project-root", Path), ("--config-name", str), ("--model-path", Path),
                 ("--output-root", Path), PORT, UI_PORT)
    backend = AceStepBackend(args.project_root, args.config_name,
                             args.model_path, args.output_root)
    launch_native_ui(backend, args.config_name, args.model_path, args.ui_port)
    serve(backend, args.port, request_path=PATH)


def launch_native_ui(backend: AceStepBackend, config_name: str,
                     model_path: Path, port: int) -> None:
    """Mount ACE-Step's own full editor using the already-loaded handlers."""
    from acestep.acestep_v15_pipeline import (
        create_demo, get_gpu_config, set_global_gpu_config,
    )

    gpu_config = get_gpu_config()
    set_global_gpu_config(gpu_config)
    demo = create_demo(init_params={
        "pre_initialized": True,
        "service_mode": False,
        "checkpoint": str(model_path),
        "config_path": config_name,
        "device": "auto",
        "init_llm": False,
        "lm_model_path": None,
        "backend": gpu_config.recommended_backend,
        "use_flash_attention": None,
        "offload_to_cpu": False,
        "offload_dit_to_cpu": False,
        "quantization": None,
        "init_status": "ACE-Step model loaded by AI-Lab",
        "enable_generate": True,
        "dit_handler": backend.dit,
        "llm_handler": backend.llm,
        "language": "en",
        "gpu_config": gpu_config,
        "output_dir": str(backend.output_root),
        "default_batch_size": 1,
    }, language="en")
    demo.queue(max_size=20, default_concurrency_limit=1)
    demo.launch(server_name=ALL_INTERFACES, server_port=port,
                share=False, inbrowser=False, prevent_thread_lock=True,
                show_error=True, allowed_paths=[str(backend.output_root)])


if __name__ == "__main__":
    main()
