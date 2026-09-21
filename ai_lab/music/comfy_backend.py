"""Workflow-based music with ComfyUI lifecycle shared with image generation."""
from __future__ import annotations

import base64
import copy
import json
import secrets
import subprocess
import tempfile
import time
import urllib.parse
import uuid
import wave
from pathlib import Path
from threading import Lock

from ai_lab.images.comfyui_server import Backend as ComfyBackend
from ai_lab.comfyui_templates import for_model


class ComfyMusicBackend:
    def __init__(self, python: str, comfyui: str, model_paths: list[str],
                 state: Path, workflow: Path, model_name: str, ffmpeg: str,
                 comfy_port: int) -> None:
        if not workflow.is_file() or not model_paths or any(
                not Path(path).is_dir() for path in model_paths):
            raise ValueError("ComfyUI music workflow or model components are absent")
        self.template = json.loads(workflow.read_text())
        if not isinstance(self.template.get("prompt"), dict):
            raise ValueError("ComfyUI music workflow has no prompt graph")
        markers = {"__AI_LAB_CAPTION__", "__AI_LAB_LYRICS__",
                   "__AI_LAB_DURATION__", "__AI_LAB_SEED__", "__AI_LAB_OUTPUT__"}
        encoded = json.dumps(self.template)
        if any(marker not in encoded for marker in markers):
            raise ValueError("ComfyUI music workflow is missing input markers")
        self.model_name = model_name
        self.ffmpeg = ffmpeg
        self.state = state
        self.lock = Lock()
        self.comfy = ComfyBackend(python, comfyui, model_paths, state,
                                  comfy_port, workflow=workflow,
                                  ui_workflow=for_model(model_name))

    def close(self) -> None:
        self.comfy.close()

    def generate(self, body: dict) -> dict:
        request = self._validate(body)
        workflow = copy.deepcopy(self.template)
        for marker, value in {
            "__AI_LAB_CAPTION__": request["prompt"],
            "__AI_LAB_LYRICS__": request["lyrics"],
            "__AI_LAB_DURATION__": request["duration"],
            "__AI_LAB_SEED__": request["seed"],
            "__AI_LAB_OUTPUT__": "ai_lab_music/" + uuid.uuid4().hex,
        }.items():
            ComfyBackend._replace(workflow, marker, value)
        with self.lock:
            queued = self.comfy.request("POST", "/prompt", workflow)
            prompt_id = queued["prompt_id"]
            deadline = time.monotonic() + 1800
            while time.monotonic() < deadline:
                history = self.comfy.request("GET", "/history/" + prompt_id)
                if prompt_id in history:
                    break
                time.sleep(.5)
            else:
                self.comfy.interrupt()
                raise TimeoutError("ComfyUI music generation timed out")
            result = history[prompt_id]
            if result.get("status", {}).get("status_str") != "success":
                raise RuntimeError("ComfyUI music workflow failed")
            outputs = [audio for node in result.get("outputs", {}).values()
                       for audio in node.get("audio", [])]
            if len(outputs) != 1:
                raise RuntimeError("ComfyUI music workflow must return one audio file")
            query = urllib.parse.urlencode({key: outputs[0][key] for key in
                                            ("filename", "subfolder", "type")
                                            if key in outputs[0]})
            flac = self.comfy.request("GET", "/view?" + query)
            output_root = (self.state / "output").resolve()
            saved = (output_root / outputs[0].get("subfolder", "") /
                     outputs[0]["filename"]).resolve()
            if saved.is_relative_to(output_root):
                saved.unlink(missing_ok=True)
        with tempfile.TemporaryDirectory(dir=self.state) as directory:
            output = Path(directory) / "result.wav"
            subprocess.run([self.ffmpeg, "-hide_banner", "-loglevel", "error",
                            "-f", "flac", "-i", "pipe:0", "-c:a", "pcm_s16le",
                            str(output)], input=flac, check=True, timeout=120)
            wav = output.read_bytes()
            with wave.open(str(output)) as audio:
                duration = audio.getnframes() / audio.getframerate()
        return {"seed": request["seed"], "duration": duration,
                "data": [{"mime_type": "audio/wav",
                          "b64_wav": base64.b64encode(wav).decode()}]}

    def _validate(self, body: dict) -> dict:
        allowed = {"model", "prompt", "lyrics", "instrumental", "duration", "seed"}
        unknown = set(body) - allowed
        if unknown:
            raise ValueError("Unknown ComfyUI music fields: " + ", ".join(sorted(unknown)))
        if body.get("model") not in (None, self.model_name):
            raise ValueError(f"this engine serves {self.model_name}")
        prompt = body.get("prompt", "")
        lyrics = body.get("lyrics", "")
        if not isinstance(prompt, str) or not 1 <= len(prompt.strip()) <= 4000:
            raise ValueError("caption must contain 1–4000 characters")
        if not isinstance(lyrics, str) or len(lyrics) > 12000:
            raise ValueError("lyrics must contain at most 12000 characters")
        if type(body.get("instrumental")) is not bool:
            raise ValueError("instrumental must be true or false")
        if not body["instrumental"] and not lyrics.strip():
            raise ValueError("lyrics are required for a vocal song")
        duration = body.get("duration", 30)
        if type(duration) not in (int, float) or not 5 <= duration <= 120:
            raise ValueError("duration must be between 5 and 120 seconds")
        seed = body.get("seed", secrets.randbelow(2**32))
        if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
            raise ValueError("seed must be a 32-bit unsigned integer")
        return {"prompt": prompt.strip(), "lyrics": lyrics.strip() or "[Instrumental]",
                "duration": duration, "seed": seed}
