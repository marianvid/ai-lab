"""Music workflow engine backed by a supervised private ComfyUI process."""
from __future__ import annotations

from pathlib import Path

from ..types import Format, ModelSet, Task
from .base import LaunchPlan, MUSIC_PATHS
from .probe import http_ok


class ComfyMusicEngine:
    id = "comfy_music"
    display_name = "ComfyUI Music"

    def __init__(self, binary: str | None = None, server: str | None = None,
                 comfyui: str | None = None,
                 ffmpeg: str | None = None,
                 model_options: dict[str, dict] | None = None) -> None:
        self.binary = binary or "python"
        self.server = server or str(Path(__file__).resolve().parents[1] /
                                    "music" / "comfy_server.py")
        self.comfyui = comfyui or ""
        self.ffmpeg = ffmpeg or "ffmpeg"
        self.model_options = dict(model_options or {})

    def formats(self) -> frozenset[Format]:
        return frozenset({Format.SAFETENSORS, Format.COMFYUI})

    def tasks(self) -> frozenset[Task]:
        return frozenset({Task.MUSIC_GENERATION})

    def params(self, task: Task = Task.MUSIC_GENERATION):
        return ()

    def supports(self, model: ModelSet) -> bool:
        return (model.format in self.formats()
                and model.task is Task.MUSIC_GENERATION
                and model.name in self.model_options)

    def plan(self, model: ModelSet, port: int, params: dict) -> LaunchPlan:
        if not self.supports(model) or params:
            raise ValueError("ComfyUI music model or settings are unsupported")
        if not self.comfyui:
            raise ValueError("ComfyUI executable must be configured")
        options = self.model_options[model.name]
        checkpoint = Path(model.entrypoint)
        model_dir = checkpoint.parent if checkpoint.name == "diffusion_models" else checkpoint
        argv = [self.binary, self.server, "--comfyui", self.comfyui,
                "--workflow", options["workflow"], "--model-name", model.name,
                "--ffmpeg", self.ffmpeg,
                "--port", str(port)]
        for subdir in options["component_subdirs"]:
            argv.extend(["--model-path", str(model_dir / subdir)])
        return LaunchPlan(argv=argv, env={"PYTHONUNBUFFERED": "1",
                                          "HF_HUB_OFFLINE": "1",
                                          "PYTHONPATH": str(Path(__file__).resolve().parents[2])},
                          web_ui=True)

    def ready(self, port: int) -> bool:
        return http_ok(port)

    def concurrency(self, params: dict) -> int:
        return 1

    def web_ui(self) -> str:
        return "native"

    def needs_mb(self, model: ModelSet, params: dict,
                 card_total_mb: float) -> float:
        return float(self.model_options[model.name]["memory_reservation_mb"])

    def api_paths(self, task: Task = Task.MUSIC_GENERATION) -> tuple[str, ...]:
        return MUSIC_PATHS if task is Task.MUSIC_GENERATION else ()
