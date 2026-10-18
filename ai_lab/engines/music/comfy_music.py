"""Music workflow engine backed by a supervised private ComfyUI process."""
from __future__ import annotations

from pathlib import Path

from ...types import Format, ModelSet, Task
from ..base import MUSIC_PATHS, LaunchPlan
from ..isolated import ConfiguredEngine, script_env
from . import music_server


class ComfyMusicEngine(ConfiguredEngine):
    """A music workflow per model (`model_options[name]["workflow"]`), run by ComfyUI."""

    id = "comfy_music"
    display_name = "ComfyUI Music"
    task = Task.MUSIC_GENERATION
    paths = MUSIC_PATHS
    script = music_server("comfy")
    served_formats = frozenset({Format.SAFETENSORS, Format.COMFYUI})
    reserves_memory = True
    location_settings = ("comfyui", "ffmpeg")

    def plan(self, model: ModelSet, port: int, params: dict) -> LaunchPlan:
        """ComfyUI with the model's components and workflow, behind the music contract."""
        options = self._options(model, params)
        if not self.locations["comfyui"]:
            raise ValueError("ComfyUI executable must be configured")
        checkpoint = Path(model.entrypoint)
        model_dir = checkpoint.parent if checkpoint.name == "diffusion_models" else checkpoint
        argv = [self.binary, self.server, "--comfyui", self.locations["comfyui"],
                "--workflow", options["workflow"], "--model-name", model.name,
                "--ffmpeg", self.locations["ffmpeg"] or "ffmpeg", "--port", str(port)]
        for subdir in options["component_subdirs"]:
            argv.extend(["--model-path", str(model_dir / subdir)])
        return LaunchPlan(argv=argv, env=script_env(), web_ui=True)

    def web_ui(self) -> str:
        """ComfyUI's own page."""
        return "native"
