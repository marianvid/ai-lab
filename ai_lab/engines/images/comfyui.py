"""Private ComfyUI adapter used by AI-Lab's named image workflows."""

from __future__ import annotations

from ...network import LOOPBACK
from ...types import Format, ModelSet, Task
from ..base import IMAGE_EDIT_PATHS, IMAGE_GENERATION_PATHS, LaunchPlan, ParamSpec, validate
from ..isolated import IsolatedEngine, weights_mb

PARAMS = (
    ParamSpec(
        "vram_mode", "VRAM mode", "choice", "normal",
        choices=("normal", "low", "cpu"), group="memory",
        help="Normal keeps the model on the GPU and refuses models larger than "
             "the available VRAM. Low VRAM explicitly enables ComfyUI weight "
             "offload. CPU keeps processing off the GPU and is much slower."),
)
# What ComfyUI is told for each VRAM mode other than normal.
_MODE_FLAGS = {"low": "--lowvram", "cpu": "--cpu"}


class ComfyUiEngine(IsolatedEngine):
    """ComfyUI for image generation and editing, one workflow per preset."""

    id = "comfyui"
    display_name = "ComfyUI"
    task = Task.IMAGE_GENERATION
    # The task classifies the checkpoint in the model library. Once the
    # adapter is running, the workflow profile decides whether a request
    # generates or edits; one ComfyUI process can serve both operations.
    paths = IMAGE_GENERATION_PATHS + IMAGE_EDIT_PATHS
    script = ("images", "comfyui_server.py")
    served_formats = frozenset({Format.COMFYUI})
    settings = PARAMS
    finds_python = True

    def __init__(self, binary: str | None = None, server: str | None = None,
                 comfyui: str | None = None,
                 model_paths: list[str] | None = None,
                 preset_workflows: dict[str, str] | None = None) -> None:
        super().__init__(binary, server)
        self.comfyui = comfyui
        self.model_paths = list(model_paths or [])
        self.preset_workflows = dict(preset_workflows or {})

    def tasks(self) -> frozenset[Task]:
        """Generating and editing images."""
        return frozenset({Task.IMAGE_GENERATION, Task.IMAGE_EDIT})

    def plan(self, model: ModelSet, port: int, params: dict) -> LaunchPlan:
        """The ComfyUI adapter for this model, with the preset's workflow if it has one."""
        settings = self._checked(model, params)
        if not self.comfyui:
            raise ValueError("ComfyUI path is not configured (engine_settings.comfyui.comfyui)")
        argv = [self.binary, self.server, "--comfyui", self.comfyui,
                "--model-root", model.entrypoint, "--name", model.name,
                "--host", LOOPBACK, "--port", str(port)]
        mode = settings["vram_mode"]
        if mode in _MODE_FLAGS:
            argv.append(_MODE_FLAGS[mode])
        for path in self.model_paths:
            argv.extend(["--extra-model-root", path])
        if workflow := self.preset_workflows.get(model.id):
            argv.extend(["--workflow", workflow])
        return LaunchPlan(
            argv=argv, env={"PYTHONUNBUFFERED": "1"}, web_ui=True,
            splits_across_cpu=mode != "normal")

    def web_ui(self) -> str:
        """ComfyUI's own page."""
        return "native"

    def needs_mb(self, model: ModelSet, params: dict, card_total_mb: float) -> float:
        """The file size in normal mode; zero (cannot say) when weights are offloaded.

        With offload enabled the full file size is not a lower bound for GPU
        memory: ComfyUI moves weights between GPU and system RAM as it works.
        Zero tells the gateway to empty the card conservatively and let the
        runtime's explicit CPU-split plan decide whether startup succeeds.
        """
        if validate(PARAMS, params)["vram_mode"] != "normal":
            return 0.0
        return weights_mb(model)
