"""Local PaddleOCR/PaddleX text-recognition pipelines."""

from __future__ import annotations

from pathlib import Path

from ..hosts.command import which
from ..types import Format, ModelSet, Task
from .base import LaunchPlan, OCR_PATHS, validate

PARAMS = ()


class PaddleOcrEngine:
    id = "paddleocr"
    display_name = "PaddleOCR"

    def __init__(self, binary: str | None = None, server: str | None = None) -> None:
        self.binary = binary or which("python") or "python"
        self.server = server or str(
            Path(__file__).resolve().parents[1] / "images" / "server.py")

    def formats(self) -> frozenset[Format]:
        return frozenset({Format.PADDLEOCR})

    def tasks(self) -> frozenset[Task]:
        return frozenset({Task.OCR})

    def params(self, task: Task = Task.OCR):
        return PARAMS if task is Task.OCR else ()

    def plan(self, model: ModelSet, port: int, params: dict) -> LaunchPlan:
        if model.format is not Format.PADDLEOCR:
            raise ValueError(f"PaddleOCR cannot load {model.format.value} models")
        if model.task is not Task.OCR:
            raise ValueError(f"PaddleOCR cannot perform {model.task.value}")
        validate(PARAMS, params)
        return LaunchPlan(argv=[self.binary, self.server,
                                "--backend", "paddleocr",
                                "--model", model.entrypoint,
                                "--name", model.name,
                                "--host", "0.0.0.0", "--port", str(port)],
                          env={"PYTHONUNBUFFERED": "1"})

    def ready(self, port: int) -> bool:
        from .probe import http_ok
        return http_ok(port)

    def concurrency(self, params: dict) -> int:
        return 1

    def needs_mb(self, model: ModelSet, params: dict,
                 card_total_mb: float) -> float:
        return model.size_bytes / (1024 * 1024)

    def api_paths(self, task: Task = Task.OCR) -> tuple[str, ...]:
        return OCR_PATHS if task is Task.OCR else ()
