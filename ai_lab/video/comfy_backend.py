"""Supervised ComfyUI video generation from an uploaded reference PNG."""
from __future__ import annotations

import base64
import binascii
import copy
import json
import secrets
import time
import uuid
from pathlib import Path
from threading import Lock

from ai_lab.images.comfyui_server import Backend as ComfyBackend
from ai_lab.comfyui_templates import for_model

MAX_IMAGE_BYTES = 25 * 1024 * 1024
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class ComfyVideoBackend:
    def __init__(self, python: str, comfyui: str, model_paths: list[str],
                 state: Path, workflow: Path, model_name: str,
                 comfy_port: int) -> None:
        if not workflow.is_file() or not model_paths or any(
                not Path(path).is_dir() for path in model_paths):
            raise ValueError("ComfyUI video workflow or model components are absent")
        self.template = json.loads(workflow.read_text())
        encoded = json.dumps(self.template)
        markers = {"__AI_LAB_PROMPT__", "__AI_LAB_INPUT__",
                   "__AI_LAB_SEED__", "__AI_LAB_OUTPUT__"}
        if not isinstance(self.template.get("prompt"), dict) or any(
                marker not in encoded for marker in markers):
            raise ValueError("ComfyUI video workflow is missing input markers")
        self.model_name = model_name
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
        with self.lock:
            filename = self.comfy._upload(request["image"])
            for marker, value in {
                "__AI_LAB_PROMPT__": request["prompt"],
                "__AI_LAB_INPUT__": filename,
                "__AI_LAB_SEED__": request["seed"],
                "__AI_LAB_OUTPUT__": "ai_lab_video/" + uuid.uuid4().hex,
            }.items():
                ComfyBackend._replace(workflow, marker, value)
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
                raise TimeoutError("ComfyUI video generation timed out")
            result = history[prompt_id]
            if result.get("status", {}).get("status_str") != "success":
                raise RuntimeError("ComfyUI video workflow failed")
            outputs = [video for node in result.get("outputs", {}).values()
                       for key in ("videos", "gifs", "images")
                       for video in node.get(key, [])
                       if str(video.get("filename", "")).lower().endswith(".mp4")]
            if len(outputs) != 1:
                raise RuntimeError("ComfyUI video workflow must return one MP4")
            output_root = (self.state / "output").resolve()
            path = (output_root / outputs[0].get("subfolder", "") /
                    outputs[0]["filename"]).resolve()
            if not path.is_relative_to(output_root) or path.suffix.lower() != ".mp4":
                raise RuntimeError("ComfyUI video output escaped the job directory")
            video = path.read_bytes()
            path.unlink(missing_ok=True)
        if len(video) < 12 or video[4:8] != b"ftyp":
            raise RuntimeError("ComfyUI did not return an MP4")
        return {"seed": request["seed"], "data": [{"mime_type": "video/mp4",
                "b64_mp4": base64.b64encode(video).decode()}]}

    def _validate(self, body: dict) -> dict:
        allowed = {"model", "prompt", "seed", "image_base64"}
        unknown = set(body) - allowed
        if unknown:
            raise ValueError("Unknown ComfyUI video fields: " + ", ".join(sorted(unknown)))
        if body.get("model") not in (None, self.model_name):
            raise ValueError(f"this engine serves {self.model_name}")
        prompt = body.get("prompt", "")
        if not isinstance(prompt, str) or not 1 <= len(prompt.strip()) <= 4000:
            raise ValueError("prompt must contain 1–4000 characters")
        encoded = body.get("image_base64", "")
        if not isinstance(encoded, str) or not encoded or len(encoded) > MAX_IMAGE_BYTES * 4 // 3 + 4:
            raise ValueError("reference PNG is missing or too large")
        try:
            image = base64.b64decode(encoded, validate=True)
        except binascii.Error:
            raise ValueError("reference image is not valid base64") from None
        if not image.startswith(PNG_SIGNATURE) or len(image) > MAX_IMAGE_BYTES:
            raise ValueError("reference image must be a PNG of at most 25 MiB")
        seed = body.get("seed", secrets.randbelow(2**32))
        if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
            raise ValueError("seed must be a 32-bit unsigned integer")
        return {"prompt": prompt.strip(), "image": image, "seed": seed}
