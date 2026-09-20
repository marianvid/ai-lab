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

    def plan(self, model: ModelSet, port: int, params: dict) -> LaunchPlan:
        if not self.supports(model):
            raise ValueError(f"Qwen3-TTS runtime is not configured for {model.name}")
        if params:
            raise ValueError("Qwen3-TTS has no instance settings")
        checkpoint = Path(model.entrypoint)
        model_directory = checkpoint.parent if checkpoint.is_file() else checkpoint
        return LaunchPlan(argv=[self.binary, self.server,
                                "--model-path", str(model_directory),
                                "--mode", self.model_modes[model.name],
                                "--port", str(port)],
                          env={"PYTHONUNBUFFERED": "1"})

    def ready(self, port: int) -> bool:
        return http_ok(port)

    def concurrency(self, params: dict) -> int:
        return 1

    def needs_mb(self, model: ModelSet, params: dict,
                 card_total_mb: float) -> float:
        return model.size_bytes / (1024 * 1024)

    def api_paths(self, task: Task = Task.SPEECH_SYNTHESIS) -> tuple[str, ...]:
        return SPEECH_PATHS if task is Task.SPEECH_SYNTHESIS else ()
