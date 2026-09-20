"""ACE-Step music engine: isolated runtime with an AI-Lab HTTP contract."""
from __future__ import annotations

from pathlib import Path

from ..types import Format, ModelSet, Task
from .base import LaunchPlan, MUSIC_PATHS
from .probe import http_ok


class AceStepEngine:
    id = "acestep"
    display_name = "ACE-Step 1.5"

    def __init__(self, binary: str | None = None, server: str | None = None,
                 project_root: str | None = None,
                 model_configs: dict[str, str] | None = None,
                 output_root: str | None = None) -> None:
        self.binary = binary or "python"
        self.server = server or str(Path(__file__).resolve().parents[1] /
                                    "music" / "server.py")
        self.project_root = project_root or ""
        self.model_configs = dict(model_configs or {})
        self.output_root = output_root or ""

    def formats(self) -> frozenset[Format]:
        return frozenset({Format.SAFETENSORS})

    def tasks(self) -> frozenset[Task]:
        return frozenset({Task.MUSIC_GENERATION})

    def params(self, task: Task = Task.MUSIC_GENERATION):
        return ()

    def supports(self, model: ModelSet) -> bool:
        return (model.format is Format.SAFETENSORS
                and model.task is Task.MUSIC_GENERATION
                and model.name in self.model_configs)

    def plan(self, model: ModelSet, port: int, params: dict) -> LaunchPlan:
        if model.format is not Format.SAFETENSORS or model.task is not Task.MUSIC_GENERATION:
            raise ValueError("ACE-Step requires a music safetensors model")
        if model.name not in self.model_configs:
            raise ValueError(f"ACE-Step runtime is not configured for {model.name}")
        if params:
            raise ValueError("ACE-Step has no instance settings")
        config_name = self.model_configs.get(model.name)
        if not config_name or not self.project_root or not self.output_root:
            raise ValueError(f"ACE-Step runtime is not configured for {model.name}")
        return LaunchPlan(argv=[
            self.binary, self.server, "--project-root", self.project_root,
            "--config-name", config_name, "--model-path", model.entrypoint,
            "--output-root", self.output_root, "--port", str(port),
        ], env={"PYTHONUNBUFFERED": "1"})

    def ready(self, port: int) -> bool:
        return http_ok(port)

    def concurrency(self, params: dict) -> int:
        return 1

    def needs_mb(self, model: ModelSet, params: dict,
                 card_total_mb: float) -> float:
        return model.size_bytes / (1024 * 1024)

    def api_paths(self, task: Task = Task.MUSIC_GENERATION) -> tuple[str, ...]:
        return MUSIC_PATHS if task is Task.MUSIC_GENERATION else ()
