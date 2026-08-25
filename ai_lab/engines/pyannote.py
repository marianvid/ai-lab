"""Local pyannote.audio speaker diarization pipelines."""

from __future__ import annotations

from pathlib import Path

from ..hosts.command import which
from ..types import Format, ModelSet, Task
from .base import DIARIZATION_PATHS, LaunchPlan, validate
from .probe import http_ok


PARAMS = ()


class PyannoteEngine:
    id = "pyannote"
    display_name = "pyannote.audio"

    def __init__(self, binary: str | None = None, server: str | None = None) -> None:
        self.binary = binary or which("python") or "python"
        self.server = server or str(
            Path(__file__).resolve().parents[1] / "audio" / "server.py")

    def formats(self) -> frozenset[Format]:
        return frozenset({Format.PYANNOTE})

    def tasks(self) -> frozenset[Task]:
        return frozenset({Task.DIARIZATION})

    def params(self, task: Task = Task.DIARIZATION) -> tuple[ParamSpec, ...]:
        return PARAMS if task is Task.DIARIZATION else ()

    def plan(self, model: ModelSet, port: int, params: dict) -> LaunchPlan:
        if model.format is not Format.PYANNOTE:
            raise ValueError(f"pyannote cannot load {model.format.value} models")
        if model.task is not Task.DIARIZATION:
            raise ValueError(f"pyannote cannot perform {model.task.value}")
        validate(PARAMS, params)
        return LaunchPlan(argv=[self.binary, self.server,
                                "--backend", "pyannote",
                                "--model", model.entrypoint,
                                "--name", model.name,
                                "--host", "0.0.0.0", "--port", str(port)],
                          env={"PYTHONUNBUFFERED": "1"})

    def ready(self, port: int) -> bool:
        return http_ok(port)

    def concurrency(self, params: dict) -> int:
        return 1

    def needs_mb(self, model: ModelSet, params: dict,
                 card_total_mb: float) -> float:
        return model.size_bytes / (1024 * 1024)

    def api_paths(self, task: Task = Task.DIARIZATION) -> tuple[str, ...]:
        return DIARIZATION_PATHS if task is Task.DIARIZATION else ()
