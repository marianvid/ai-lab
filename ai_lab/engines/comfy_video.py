"""Configurable image-to-video workflows in an isolated ComfyUI process."""
from __future__ import annotations

from pathlib import Path

from ..types import Format, ModelSet, Task
from .base import LaunchPlan, VIDEO_PATHS
from .probe import http_ok


class ComfyVideoEngine:
    id = "comfy_video"
    display_name = "ComfyUI Video"

    def __init__(self, binary: str | None = None, server: str | None = None,
                 comfyui: str | None = None,
                 model_options: dict[str, dict] | None = None) -> None:
        self.binary = binary or "python"
        self.server = server or str(Path(__file__).resolve().parents[1] /
                                    "video" / "comfy_server.py")
        self.comfyui = comfyui or ""
        self.model_options = dict(model_options or {})

    def formats(self) -> frozenset[Format]:
        return frozenset({Format.COMFYUI})

    def tasks(self) -> frozenset[Task]:
        return frozenset({Task.VIDEO_GENERATION})

    def params(self, task: Task = Task.VIDEO_GENERATION):
        return ()

    def supports(self, model: ModelSet) -> bool:
        return (model.format is Format.COMFYUI
                and model.task is Task.VIDEO_GENERATION
                and model.name in self.model_options)

    def video_form(self, model_name: str) -> dict:
        options = self.model_options[model_name]
        return {"image_required": True,
                "clip_seconds": options["clip_seconds"]}

    def plan(self, model: ModelSet, port: int, params: dict) -> LaunchPlan:
        if not self.supports(model) or params:
            raise ValueError("ComfyUI video model or settings are unsupported")
        if not self.comfyui:
            raise ValueError("ComfyUI executable must be configured")
        options = self.model_options[model.name]
        root = Path(options["model_root"])
        if not Path(model.entrypoint).is_relative_to(root):
            raise ValueError("ComfyUI video root does not match the model")
        argv = [self.binary, self.server, "--comfyui", self.comfyui,
                "--workflow", options["workflow"],
                "--model-name", model.name, "--port", str(port)]
        for subdir in options["component_subdirs"]:
            argv.extend(["--model-path", str(root / subdir)])
        return LaunchPlan(argv=argv, env={"PYTHONUNBUFFERED": "1",
                                          "HF_HUB_OFFLINE": "1",
                                          "PYTHONPATH": str(Path(__file__).resolve().parents[2])},
                          splits_across_cpu=True, web_ui=True)

    def ready(self, port: int) -> bool:
        return http_ok(port)

    def concurrency(self, params: dict) -> int:
        return 1

    def web_ui(self) -> str:
        return "native"

    def needs_mb(self, model: ModelSet, params: dict,
                 card_total_mb: float) -> float:
        return float(self.model_options[model.name]["memory_reservation_mb"])

    def api_paths(self, task: Task = Task.VIDEO_GENERATION) -> tuple[str, ...]:
        return VIDEO_PATHS if task is Task.VIDEO_GENERATION else ()
