"""Whisper transcription accelerated by MLX on Apple silicon."""

from __future__ import annotations

from pathlib import Path

from ..hosts.command import which
from ..types import Format, ModelSet, Task
from .base import LaunchPlan, TRANSCRIPTION_PATHS
from .probe import http_ok


class MlxWhisperEngine:
    id = "mlxwhisper"
    display_name = "MLX Whisper"

    def __init__(self, binary: str | None = None,
                 server: str | None = None) -> None:
        self.binary = binary or which("python") or "python"
        self.server = server or str(
            Path(__file__).resolve().parents[1] / "audio" / "server.py")

    def formats(self) -> frozenset[Format]:
        return frozenset({Format.SAFETENSORS})

    def tasks(self) -> frozenset[Task]:
        return frozenset({Task.TRANSCRIPTION})

    def params(self, task: Task = Task.TRANSCRIPTION):
        return ()

    def plan(self, model: ModelSet, port: int, params: dict) -> LaunchPlan:
        if model.format is not Format.SAFETENSORS:
            raise ValueError(f"MLX Whisper cannot load {model.format.value} models")
        if model.task is not Task.TRANSCRIPTION:
            raise ValueError(f"MLX Whisper cannot perform {model.task.value}")
        if params:
            raise ValueError(f"Unknown settings: {', '.join(sorted(params))}")
        return LaunchPlan(
            argv=[self.binary, self.server, "--backend", "mlx-whisper",
                  "--model", model.entrypoint, "--name", model.name,
                  "--host", "0.0.0.0", "--port", str(port)],
            env={"PYTHONUNBUFFERED": "1"},
        )

    def ready(self, port: int) -> bool:
        return http_ok(port)

    def concurrency(self, params: dict) -> int:
        return 1

    def needs_mb(self, model: ModelSet, params: dict,
                 card_total_mb: float) -> float:
        return model.size_bytes / (1024 * 1024)

    def api_paths(self, task: Task = Task.TRANSCRIPTION) -> tuple[str, ...]:
        return TRANSCRIPTION_PATHS if task is Task.TRANSCRIPTION else ()
