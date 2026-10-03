"""LeVo 2 (SongGeneration 2): songs or instrumentals, research licence only."""
from __future__ import annotations

from pathlib import Path

from ..types import Format, ModelSet, Task
from .base import LaunchPlan, MUSIC_PATHS
from .probe import http_ok


class Levo2Engine:
    id = "levo2"
    display_name = "LeVo 2"

    def __init__(self, binary: str | None = None, server: str | None = None,
                 cantor: str | None = None, output_root: str | None = None,
                 model_options: dict[str, dict] | None = None) -> None:
        # `binary` is the Python that runs the small HTTP host; `cantor` is
        # the native LeVo2.cpp program that makes the song.
        self.binary = binary or "python"
        self.server = server or str(Path(__file__).resolve().parents[1] /
                                    "music" / "levo2_server.py")
        self.cantor = cantor or ""
        self.output_root = output_root or ""
        self.model_options = dict(model_options or {})

    def formats(self) -> frozenset[Format]:
        return frozenset({Format.SAFETENSORS})

    def tasks(self) -> frozenset[Task]:
        return frozenset({Task.MUSIC_GENERATION})

    def params(self, task: Task = Task.MUSIC_GENERATION):
        return ()

    def supports(self, model: ModelSet) -> bool:
        return (model.task is Task.MUSIC_GENERATION
                and model.name in self.model_options)

    def music_form(self, model_name: str) -> dict:
        return {"lyrics_required": False, "duration_kind": "seconds"}

    def plan(self, model: ModelSet, port: int, params: dict) -> LaunchPlan:
        if not self.supports(model):
            raise ValueError(f"LeVo 2 is not configured for {model.name}")
        if params:
            raise ValueError("LeVo 2 has no instance settings")
        if not self.cantor or not self.output_root:
            raise ValueError("LeVo 2 runtime and output paths must be configured")
        options = self.model_options[model.name]
        entry = Path(model.entrypoint)
        model_dir = entry.parent if entry.is_file() else entry
        return LaunchPlan(argv=[
            self.binary, self.server, "--cantor", self.cantor,
            "--model-path", str(model_dir), "--lm", options["lm"],
            "--flow", options["flow"], "--vae", options["vae"],
            "--output-root", self.output_root,
            "--steps", str(options["steps"]), "--cfg", str(options["cfg"]),
            "--port", str(port)],
            env={"PYTHONUNBUFFERED": "1",
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
