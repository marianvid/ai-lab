"""Workflow-based music with ComfyUI lifecycle shared with image generation."""
from __future__ import annotations

import copy
import json
import subprocess
import tempfile
import time
import urllib.parse
import uuid
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

from ai_lab.comfyui_templates import for_model
from ai_lab.images.comfyui_server import Backend as ComfyBackend
from ai_lab.images.comfyui_server import ComfyLaunch, replace_marker

from .request_fields import (
    MAX_LYRICS_CHARS,
    MAX_PROMPT_CHARS,
    TextRule,
    check_fields,
    flag,
    seconds,
    seed,
    text,
    wav_answer,
    wav_length_s,
)

ALLOWED_FIELDS = {"model", "prompt", "lyrics", "instrumental", "duration", "seed"}
PROMPT = TextRule("caption must contain 1–4000 characters", MAX_PROMPT_CHARS, shortest=1)
LYRICS = TextRule("lyrics must contain at most 12000 characters", MAX_LYRICS_CHARS)
DEFAULT_S, SHORTEST_S, LONGEST_S = 30, 5, 120
MARKERS = {"__AI_LAB_CAPTION__", "__AI_LAB_LYRICS__", "__AI_LAB_DURATION__",
           "__AI_LAB_SEED__", "__AI_LAB_OUTPUT__"}
# How long one song may take in ComfyUI, and how often it is asked.
SONG_LIMIT_S = 1800
POLL_S = 0.5
# How long ffmpeg may take to turn ComfyUI's FLAC into WAV.
CONVERT_S = 120


@dataclass(frozen=True)
class ComfyRuntime:
    """The private ComfyUI process this backend starts: where it is, its models, its port.

    `python` runs ComfyUI from the `comfyui` folder; `model_paths` are the
    folders it may load models from; `state` is its scratch folder.
    """

    python: str
    comfyui: str
    model_paths: list[str]
    state: Path
    port: int


class ComfyMusicBackend:
    """A music workflow (MiniMax-style) run by a private ComfyUI, one song at a time."""

    def __init__(self, runtime: ComfyRuntime, workflow: Path, model_name: str,
                 ffmpeg: str) -> None:
        if not workflow.is_file() or not runtime.model_paths or any(
                not Path(path).is_dir() for path in runtime.model_paths):
            raise ValueError("ComfyUI music workflow or model components are absent")
        self.template = json.loads(workflow.read_text())
        if not isinstance(self.template.get("prompt"), dict):
            raise ValueError("ComfyUI music workflow has no prompt graph")
        encoded = json.dumps(self.template)
        if any(marker not in encoded for marker in MARKERS):
            raise ValueError("ComfyUI music workflow is missing input markers")
        self.model_name = model_name
        self.ffmpeg = ffmpeg
        self.state = runtime.state
        self.lock = Lock()
        self.comfy = ComfyBackend(ComfyLaunch(
            runtime.python, runtime.comfyui, runtime.model_paths, runtime.state, runtime.port,
            workflow=workflow, ui_workflow=for_model(model_name)))

    def close(self) -> None:
        """Stop the private ComfyUI."""
        self.comfy.close()

    def generate(self, body: dict) -> dict:
        """One song for the request, as base64 WAV."""
        request = self._validate(body)
        workflow = copy.deepcopy(self.template)
        for marker, value in {
            "__AI_LAB_CAPTION__": request["prompt"],
            "__AI_LAB_LYRICS__": request["lyrics"],
            "__AI_LAB_DURATION__": request["duration"],
            "__AI_LAB_SEED__": request["seed"],
            "__AI_LAB_OUTPUT__": "ai_lab_music/" + uuid.uuid4().hex,
        }.items():
            replace_marker(workflow, marker, value)
        with self.lock:
            flac = self._fetch(self._run(workflow))
        wav, length = self._to_wav(flac)
        return wav_answer(wav, seed=request["seed"], duration=length)

    def _run(self, workflow: dict) -> dict:
        """Queue the workflow and wait for its one audio output; its description."""
        prompt_id = self._json("POST", "/prompt", workflow)["prompt_id"]
        deadline = time.monotonic() + SONG_LIMIT_S
        while time.monotonic() < deadline:
            history = self._json("GET", "/history/" + prompt_id)
            if prompt_id in history:
                break
            time.sleep(POLL_S)
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
        return outputs[0]

    def _fetch(self, output: dict) -> bytes:
        """The output's FLAC bytes; ComfyUI's own copy is deleted afterwards."""
        query = urllib.parse.urlencode({key: output[key] for key in
                                        ("filename", "subfolder", "type") if key in output})
        flac = self.comfy.request("GET", "/view?" + query)
        if not isinstance(flac, bytes):
            raise RuntimeError("ComfyUI answered the audio download with JSON, not audio")
        output_root = (self.state / "output").resolve()
        saved = (output_root / output.get("subfolder", "") / output["filename"]).resolve()
        if saved.is_relative_to(output_root):
            saved.unlink(missing_ok=True)
        return flac

    def _to_wav(self, flac: bytes) -> tuple[bytes, float]:
        """FLAC → 16-bit WAV with ffmpeg; the bytes and how long they play."""
        with tempfile.TemporaryDirectory(dir=self.state) as directory:
            output = Path(directory) / "result.wav"
            subprocess.run([self.ffmpeg, "-hide_banner", "-loglevel", "error",
                            "-f", "flac", "-i", "pipe:0", "-c:a", "pcm_s16le",
                            str(output)], input=flac, check=True, timeout=CONVERT_S)
            return output.read_bytes(), wav_length_s(output)

    def _json(self, method: str, path: str, payload: dict | None = None) -> dict:
        """A ComfyUI call that must answer JSON."""
        answer = self.comfy.request(method, path, payload)
        if not isinstance(answer, dict):
            raise RuntimeError(f"ComfyUI answered {path} with something other than JSON")
        return answer

    def _validate(self, body: dict) -> dict:
        check_fields(body, ALLOWED_FIELDS, "ComfyUI music", self.model_name)
        prompt = text(body, "prompt", PROMPT)
        lyrics = text(body, "lyrics", LYRICS)
        if type(body.get("instrumental")) is not bool:
            raise ValueError("instrumental must be true or false")
        if not flag(body, "instrumental", False) and not lyrics:
            raise ValueError("lyrics are required for a vocal song")
        return {"prompt": prompt, "lyrics": lyrics or "[Instrumental]",
                "duration": seconds(body, DEFAULT_S, SHORTEST_S, LONGEST_S),
                "seed": seed(body)}
