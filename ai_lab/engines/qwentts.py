"""Qwen3-TTS runtime for VoiceDesign and CustomVoice checkpoints."""
from __future__ import annotations

from pathlib import Path

from ..types import Format, ModelSet, Task
from .base import LaunchPlan, SPEECH_PATHS
from .probe import http_ok


class QwenTtsEngine:
    id = "qwentts"
    display_name = "Qwen3-TTS"

    def __init__(self, binary: str | None = None, server: str | None = None,
                 model_modes: dict[str, str] | None = None) -> None:
        self.binary = binary or "python"
        self.server = server or str(Path(__file__).resolve().parents[1] /
                                    "speech" / "server.py")
        self.model_modes = dict(model_modes or {})

    def formats(self) -> frozenset[Format]:
        return frozenset({Format.SAFETENSORS})

    def tasks(self) -> frozenset[Task]:
        return frozenset({Task.SPEECH_SYNTHESIS})

    def params(self, task: Task = Task.SPEECH_SYNTHESIS):
        return ()

    def supports(self, model: ModelSet) -> bool:
        return (model.format is Format.SAFETENSORS
                and model.task is Task.SPEECH_SYNTHESIS
                and self.model_modes.get(model.name) in
                {"voice-design", "custom-voice"})

    def speech_form(self, model_name: str) -> dict:
        mode = self.model_modes.get(model_name)
        return {"instruction_required": mode == "voice-design",
                "instruction_visible": True,
                "speaker_visible": mode == "custom-voice",
                "language_visible": True,
                "speaker_label": "Speaker",
                "seed_supported": True, "reference_supported": False}

    def plan(self, model: ModelSet, port: int, params: dict) -> LaunchPlan:
        if not self.supports(model):
            raise ValueError(f"Qwen3-TTS runtime is not configured for {model.name}")
        if params:
            raise ValueError("Qwen3-TTS has no instance settings")
        checkpoint = Path(model.entrypoint)
        model_directory = checkpoint.parent if checkpoint.is_file() else checkpoint
        return LaunchPlan(argv=[self.binary, self.server, "--backend", "qwen",
                                "--model-path", str(model_directory),
                                "--mode", self.model_modes[model.name],
                                "--ui-port", str(port + 10000),
                                "--port", str(port)],
                          env={"PYTHONUNBUFFERED": "1",
                               "PYTHONPATH": str(Path(__file__).resolve().parents[2])})

    def ready(self, port: int) -> bool:
        return http_ok(port)

    def concurrency(self, params: dict) -> int:
        return 1

    def needs_mb(self, model: ModelSet, params: dict,
                 card_total_mb: float) -> float:
        weights_mb = model.size_bytes / (1024 * 1024)
        # The runtime also holds activations, attention buffers and CUDA
        # allocations. Weight bytes alone let the scheduler admit a second
        # checkpoint into less free VRAM than loading actually requires.
        return max(weights_mb * 1.5, weights_mb + 2048)

    def api_paths(self, task: Task = Task.SPEECH_SYNTHESIS) -> tuple[str, ...]:
        return SPEECH_PATHS if task is Task.SPEECH_SYNTHESIS else ()
