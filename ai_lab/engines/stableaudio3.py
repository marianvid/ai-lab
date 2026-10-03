"""Stable Audio 3: instrumental music and sound from a text prompt."""
from __future__ import annotations

from pathlib import Path

from ..types import Format, ModelSet, Task
from .base import LaunchPlan, MUSIC_PATHS
from .probe import http_ok


class StableAudio3Engine:
    id = "stableaudio3"
    display_name = "Stable Audio 3"

    def __init__(self, binary: str | None = None, server: str | None = None,
                 model_options: dict[str, dict] | None = None) -> None:
        self.binary = binary or "python"
        self.server = server or str(Path(__file__).resolve().parents[1] /
                                    "music" / "stableaudio3_server.py")
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
        return {"lyrics_required": False, "duration_kind": "seconds",
                "instrumental_only": True}

    def plan(self, model: ModelSet, port: int, params: dict) -> LaunchPlan:
        if not self.supports(model):
            raise ValueError(f"Stable Audio 3 is not configured for {model.name}")
        if params:
            raise ValueError("Stable Audio 3 has no instance settings")
        options = self.model_options[model.name]
        entry = Path(model.entrypoint)
        model_dir = entry.parent if entry.is_file() else entry
        return LaunchPlan(argv=[
            self.binary, self.server, "--model-path", str(model_dir),
            "--steps", str(options["steps"]),
            "--cfg-scale", str(options["cfg_scale"]), "--port", str(port),
            "--ui-port", str(port + 10000)],
            env={"PYTHONUNBUFFERED": "1", "HF_HUB_OFFLINE": "1",
                 "GRADIO_ANALYTICS_ENABLED": "False",
                 "PYTHONPATH": str(Path(__file__).resolve().parents[2])},
            web_ui=True)

    def web_ui(self) -> str:
        # Stability's official Gradio page, on the instance port + 10000.
        return "native"

    def ready(self, port: int) -> bool:
        return http_ok(port)

    def concurrency(self, params: dict) -> int:
        return 1

    def needs_mb(self, model: ModelSet, params: dict,
                 card_total_mb: float) -> float:
        return float(self.model_options[model.name]["memory_reservation_mb"])

    def api_paths(self, task: Task = Task.MUSIC_GENERATION) -> tuple[str, ...]:
        return MUSIC_PATHS if task is Task.MUSIC_GENERATION else ()
