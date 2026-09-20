"""HeartMuLa lyric-conditioned music from a configurable local checkpoint."""
from __future__ import annotations

from pathlib import Path

from ..types import Format, ModelSet, Task
from .base import LaunchPlan, MUSIC_PATHS
from .probe import http_ok


class HeartMulaEngine:
    id = "heartmula"
    display_name = "HeartMuLa"

    def __init__(self, binary: str | None = None, server: str | None = None,
                 bundle_root: str | None = None,
                 output_root: str | None = None,
                 model_options: dict[str, dict] | None = None) -> None:
        self.binary = binary or "python"
        self.server = server or str(Path(__file__).resolve().parents[1] /
                                    "music" / "heartmula_server.py")
        self.output_root = output_root or ""
        self.bundle_root = bundle_root or ""
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
        return {"lyrics_required": True}

    def plan(self, model: ModelSet, port: int, params: dict) -> LaunchPlan:
        if not self.supports(model):
            raise ValueError(f"HeartMuLa is not configured for {model.name}")
        if params:
            raise ValueError("HeartMuLa has no instance settings")
        if not self.output_root or not self.bundle_root:
            raise ValueError("HeartMuLa bundle and output paths must be configured")
        options = self.model_options[model.name]
        checkpoint = Path(model.entrypoint)
        model_dir = checkpoint.parent if checkpoint.is_file() else checkpoint
        return LaunchPlan(argv=[
            self.binary, self.server, "--bundle-root", self.bundle_root,
            "--checkpoint-path", str(model_dir),
            "--checkpoint-subdir", options["checkpoint_subdir"],
            "--output-root", self.output_root,
            "--version", options["version"],
            "--topk", str(options["topk"]),
            "--temperature", str(options["temperature"]),
            "--cfg-scale", str(options["cfg_scale"]),
            "--port", str(port)],
            env={"PYTHONUNBUFFERED": "1", "HF_HUB_OFFLINE": "1",
                 "PYTHONPATH": str(Path(__file__).resolve().parents[2])})

    def ready(self, port: int) -> bool:
        return http_ok(port)

    def concurrency(self, params: dict) -> int:
        return 1

    def needs_mb(self, model: ModelSet, params: dict,
                 card_total_mb: float) -> float:
        return float(self.model_options[model.name]["memory_reservation_mb"])

    def api_paths(self, task: Task = Task.MUSIC_GENERATION) -> tuple[str, ...]:
        return MUSIC_PATHS if task is Task.MUSIC_GENERATION else ()
