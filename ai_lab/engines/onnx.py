"""ONNX Runtime for small speech models that do not need the GPU."""

from __future__ import annotations

from pathlib import Path

from ..hosts.command import which
from ..types import Format, ModelSet, Task
from .base import LaunchPlan, VAD_PATHS
from .probe import http_ok


class OnnxEngine:
    id = "onnx"
    display_name = "ONNX Runtime"

    def __init__(self, binary: str | None = None,
                 server: str | None = None) -> None:
        self.binary = binary or which("python") or "python"
        self.server = server or str(
            Path(__file__).resolve().parents[1] / "audio" / "server.py")

    def formats(self) -> frozenset[Format]:
        return frozenset({Format.ONNX})

    def tasks(self) -> frozenset[Task]:
        return frozenset({Task.VAD})

    def params(self, task: Task = Task.VAD):
        return ()

    def plan(self, model: ModelSet, port: int, params: dict) -> LaunchPlan:
        if model.format is not Format.ONNX:
            raise ValueError(f"ONNX Runtime cannot load {model.format.value} models")
        if model.task is not Task.VAD:
            raise ValueError(f"ONNX Runtime cannot perform {model.task.value}")
        if params:
            raise ValueError(f"Unknown settings: {', '.join(sorted(params))}")
        return LaunchPlan(
            argv=[self.binary, self.server,
                  "--backend", "silero",
                  "--model", model.entrypoint,
                  "--name", model.name,
                  "--host", "0.0.0.0", "--port", str(port)],
            env={"PYTHONUNBUFFERED": "1"},
        )

    def ready(self, port: int) -> bool:
        return http_ok(port)

    def concurrency(self, params: dict) -> int:
        return 1

    def needs_mb(self, model: ModelSet, params: dict,
                 card_total_mb: float) -> float:
        return 0.0

    def api_paths(self, task: Task = Task.VAD) -> tuple[str, ...]:
        return VAD_PATHS if task is Task.VAD else ()
