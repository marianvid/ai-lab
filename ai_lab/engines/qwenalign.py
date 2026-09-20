"""Qwen forced alignment with local weights and the shared audio gateway."""
from __future__ import annotations

from pathlib import Path

from ..types import Format, ModelSet, Task
from .base import ALIGNMENT_PATHS, LaunchPlan
from .probe import http_ok


class QwenAlignEngine:
    id = "qwenalign"
    display_name = "Qwen Forced Aligner"

    def __init__(self, binary: str | None = None,
                 server: str | None = None) -> None:
        self.binary = binary or "python"
        self.server = server or str(Path(__file__).resolve().parents[1] /
                                    "audio" / "server.py")

    def formats(self) -> frozenset[Format]:
        return frozenset({Format.SAFETENSORS})

    def tasks(self) -> frozenset[Task]:
        return frozenset({Task.ALIGNMENT})

    def params(self, task: Task = Task.ALIGNMENT):
        return ()

    def plan(self, model: ModelSet, port: int, params: dict) -> LaunchPlan:
        if model.format is not Format.SAFETENSORS or model.task is not Task.ALIGNMENT:
            raise ValueError("Qwen aligner requires an alignment checkpoint")
        if params:
            raise ValueError("Qwen aligner has no instance settings")
        checkpoint = Path(model.entrypoint)
        model_dir = checkpoint.parent if checkpoint.is_file() else checkpoint
        return LaunchPlan(argv=[
            self.binary, self.server, "--backend", "qwen-align",
            "--model", str(model_dir), "--name", model.name,
            "--host", "0.0.0.0", "--port", str(port)],
            env={"PYTHONUNBUFFERED": "1", "HF_HUB_OFFLINE": "1",
                 "PYTHONPATH": str(Path(__file__).resolve().parents[2])})

    def ready(self, port: int) -> bool:
        return http_ok(port)

    def concurrency(self, params: dict) -> int:
        return 1

    def needs_mb(self, model: ModelSet, params: dict,
                 card_total_mb: float) -> float:
        weights_mb = model.size_bytes / (1024 * 1024)
        return max(weights_mb * 1.5, weights_mb + 1024)

    def api_paths(self, task: Task = Task.ALIGNMENT) -> tuple[str, ...]:
        return ALIGNMENT_PATHS if task is Task.ALIGNMENT else ()
