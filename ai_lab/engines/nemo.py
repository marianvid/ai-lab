"""NVIDIA NeMo Speech models, served behind AI-Lab's common audio API."""

from __future__ import annotations

from pathlib import Path

from ..hosts.command import which
from ..types import Format, ModelSet, Task
from .base import (DIARIZATION_PATHS, LaunchPlan, ParamSpec,
                   TRANSCRIPTION_PATHS, validate)
from .probe import http_ok


PARAMS = (
    ParamSpec("precision", "Precision", "choice", "bf16",
              choices=("bf16", "fp16", "fp32"), group="memory",
              help="Arithmetic used by the speech model. BF16 is the native "
                   "choice on this Blackwell card; FP32 uses roughly twice "
                   "the memory and is mainly a diagnostic setting."),
)


class NemoEngine:
    id = "nemo"
    display_name = "NVIDIA NeMo Speech"

    def __init__(self, binary: str | None = None,
                 server: str | None = None) -> None:
        self.binary = binary or which("python") or "python"
        self.server = server or str(
            Path(__file__).resolve().parents[1] / "audio" / "server.py")

    def formats(self) -> frozenset[Format]:
        return frozenset({Format.NEMO})

    def tasks(self) -> frozenset[Task]:
        return frozenset({Task.TRANSCRIPTION, Task.DIARIZATION})

    def params(self, task: Task = Task.TRANSCRIPTION) -> tuple[ParamSpec, ...]:
        return PARAMS if task in (Task.TRANSCRIPTION, Task.DIARIZATION) else ()

    def plan(self, model: ModelSet, port: int, params: dict) -> LaunchPlan:
        if model.format is not Format.NEMO:
            raise ValueError(f"NeMo cannot load {model.format.value} models")
        if model.task not in self.tasks():
            raise ValueError(f"NeMo cannot perform {model.task.value}")
        if not model.complete:
            raise ValueError(f"{model.name} is incomplete")
        settings = validate(PARAMS, params)
        return LaunchPlan(
            argv=[self.binary, self.server,
                  "--backend", ("sortformer" if model.task is Task.DIARIZATION
                                else "nemo"),
                  "--model", model.entrypoint,
                  "--name", model.name,
                  "--host", "0.0.0.0",
                  "--port", str(port),
                  "--precision", str(settings["precision"])],
            env={"PYTHONUNBUFFERED": "1"},
        )

    def ready(self, port: int) -> bool:
        return http_ok(port)

    def concurrency(self, params: dict) -> int:
        return 1

    def needs_mb(self, model: ModelSet, params: dict,
                 card_total_mb: float) -> float:
        multiplier = 2.0 if validate(PARAMS, params)["precision"] == "fp32" else 1.0
        return model.size_bytes / (1024 * 1024) * multiplier

    def api_paths(self, task: Task = Task.TRANSCRIPTION) -> tuple[str, ...]:
        if task is Task.TRANSCRIPTION:
            return TRANSCRIPTION_PATHS
        if task is Task.DIARIZATION:
            return DIARIZATION_PATHS
        return ()
