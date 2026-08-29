"""Private ComfyUI adapter used by AI-Lab's named image workflows."""

from __future__ import annotations

from pathlib import Path

from ..hosts.command import which
from ..types import Format, ModelSet, Task
from .base import (IMAGE_EDIT_PATHS, IMAGE_GENERATION_PATHS, LaunchPlan,
                   ParamSpec, validate)

PARAMS = (
    ParamSpec(
        "vram_mode", "VRAM mode", "choice", "normal",
        choices=("normal", "low", "cpu"), group="memory",
        help="Normal keeps the model on the GPU and refuses models larger than "
             "the available VRAM. Low VRAM explicitly enables ComfyUI weight "
             "offload. CPU keeps processing off the GPU and is much slower."),
)


class ComfyUiEngine:
    id = "comfyui"
    display_name = "ComfyUI"

    def __init__(self, binary: str | None = None, server: str | None = None,
                 comfyui: str | None = None,
                 model_paths: list[str] | None = None) -> None:
        self.binary = binary or which("python") or "python"
        self.server = server or str(
            Path(__file__).resolve().parents[1] / "images" / "comfyui_server.py")
        self.comfyui = comfyui or "/opt/ComfyUI/main.py"
        self.model_paths = list(model_paths or [])

    def formats(self) -> frozenset[Format]:
        return frozenset({Format.COMFYUI})

    def tasks(self) -> frozenset[Task]:
        return frozenset({Task.IMAGE_GENERATION, Task.IMAGE_EDIT})

    def params(self, task: Task = Task.IMAGE_GENERATION):
        return PARAMS if task in self.tasks() else ()

    def plan(self, model: ModelSet, port: int, params: dict) -> LaunchPlan:
        if model.format is not Format.COMFYUI:
            raise ValueError(f"ComfyUI cannot load {model.format.value} models")
        if model.task not in self.tasks():
            raise ValueError(f"ComfyUI cannot perform {model.task.value}")
        settings = validate(PARAMS, params)
        argv = [self.binary, self.server, "--comfyui", self.comfyui,
                  "--model-root", model.entrypoint, "--name", model.name,
                  "--host", "0.0.0.0", "--port", str(port)]
        mode = settings["vram_mode"]
        if mode == "low":
            argv.append("--lowvram")
        elif mode == "cpu":
            argv.append("--cpu")
        for path in self.model_paths:
            argv.extend(["--extra-model-root", path])
        return LaunchPlan(
            argv=argv, env={"PYTHONUNBUFFERED": "1"}, web_ui=False,
            splits_across_cpu=mode != "normal")

    def ready(self, port: int) -> bool:
        from .probe import http_ok
        return http_ok(port)

    def concurrency(self, params: dict) -> int:
        # ComfyUI's interrupt endpoint targets the current global execution.
        return 1

    def needs_mb(self, model: ModelSet, params: dict,
                 card_total_mb: float) -> float:
        return model.size_bytes / (1024 * 1024)

    def api_paths(self, task: Task = Task.IMAGE_GENERATION) -> tuple[str, ...]:
        # The task classifies the checkpoint in the model library. Once the
        # adapter is running, the workflow profile decides whether a request
        # generates or edits; one ComfyUI process can serve both operations.
        if task in self.tasks():
            return IMAGE_GENERATION_PATHS + IMAGE_EDIT_PATHS
        return ()
