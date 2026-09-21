"""Khala music via the isolated Apple Silicon vanilla generator."""
from __future__ import annotations

from pathlib import Path

from ..types import Format, ModelSet, Task
from .base import LaunchPlan, MUSIC_PATHS
from .probe import http_ok


class KhalaEngine:
    id = "khala"
    display_name = "Khala"

    def __init__(self, binary: str | None = None, server: str | None = None,
                 generator_script: str | None = None,
                 output_root: str | None = None,
                 model_options: dict[str, dict] | None = None) -> None:
        self.binary = binary or "python"
        self.server = server or str(Path(__file__).resolve().parents[1] /
                                    "music" / "khala_server.py")
        self.generator_script = generator_script or ""
        self.output_root = output_root or ""
        self.model_options = dict(model_options or {})

    def formats(self) -> frozenset[Format]:
        return frozenset({Format.SAFETENSORS})

    def tasks(self) -> frozenset[Task]:
        return frozenset({Task.MUSIC_GENERATION})

    def params(self, task: Task = Task.MUSIC_GENERATION):
        return ()

    def supports(self, model: ModelSet) -> bool:
        return (model.format is Format.SAFETENSORS
                and model.task is Task.MUSIC_GENERATION
                and model.name in self.model_options)

    def music_form(self, model_name: str) -> dict:
        options = self.model_options.get(model_name, {})
        return {"duration_kind": "bucket", "default_bucket":
                options.get("default_bucket", 0), "maximum_bucket":
                options.get("maximum_bucket", 4)}

    def plan(self, model: ModelSet, port: int, params: dict) -> LaunchPlan:
        if not self.supports(model):
            raise ValueError(f"Khala is not configured for {model.name}")
        if params:
            raise ValueError("Khala has no instance settings")
        if not self.generator_script or not self.output_root:
            raise ValueError("Khala generator and output path must be configured")
        options = self.model_options[model.name]
        checkpoint = Path(model.entrypoint)
        model_dir = checkpoint.parent if checkpoint.is_file() else checkpoint
        return LaunchPlan(argv=[
            self.binary, self.server,
            "--generator-script", self.generator_script,
            "--model-path", str(model_dir),
            "--output-root", self.output_root,
            "--default-bucket", str(options.get("default_bucket", 0)),
            "--maximum-bucket", str(options.get("maximum_bucket", 4)),
            "--port", str(port), "--ui-port", str(port + 10000)],
            env={"PYTHONUNBUFFERED": "1", "HF_HUB_OFFLINE": "1",
                 "PYTHONPATH": str(Path(__file__).resolve().parents[2])})

    def ready(self, port: int) -> bool:
        return http_ok(port)

    def concurrency(self, params: dict) -> int:
        return 1

    def needs_mb(self, model: ModelSet, params: dict,
                 card_total_mb: float) -> float:
        weights_mb = model.size_bytes / (1024 * 1024)
        return max(weights_mb * 1.5, weights_mb + 2048)

    def api_paths(self, task: Task = Task.MUSIC_GENERATION) -> tuple[str, ...]:
        return MUSIC_PATHS if task is Task.MUSIC_GENERATION else ()
