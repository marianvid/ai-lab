"""LeVo 2 (SongGeneration 2): songs or instrumentals, research licence only."""
from __future__ import annotations

from ..types import ModelSet, Task
from .base import MUSIC_PATHS, LaunchPlan
from .isolated import ConfiguredEngine, model_directory, script_env


class Levo2Engine(ConfiguredEngine):
    """LeVo 2 through the native LeVo2.cpp program (`cantor`)."""

    id = "levo2"
    display_name = "LeVo 2"
    task = Task.MUSIC_GENERATION
    paths = MUSIC_PATHS
    script = ("music", "levo2_server.py")
    reserves_memory = True
    # `binary` is the Python that runs the small HTTP host; `cantor` is the
    # native LeVo2.cpp program that makes the song.
    location_settings = ("cantor", "output_root")

    def supports(self, model: ModelSet) -> bool:
        """Whether the configuration names this music model (its files are GGUF or safetensors)."""
        return model.task is self.task and self.configured(model.name)

    def music_form(self, model_name: str) -> dict:
        """What the music page shows: optional lyrics, a length in seconds."""
        return {"lyrics_required": False, "duration_kind": "seconds"}

    def plan(self, model: ModelSet, port: int, params: dict) -> LaunchPlan:
        """The LeVo 2 host for this model's language, flow and VAE files."""
        options = self._options(
            model, params, "LeVo 2 runtime and output paths must be configured")
        return LaunchPlan(argv=[
            self.binary, self.server, "--cantor", self.locations["cantor"],
            "--model-path", model_directory(model), "--lm", options["lm"],
            "--flow", options["flow"], "--vae", options["vae"],
            "--output-root", self.locations["output_root"],
            "--steps", str(options["steps"]), "--cfg", str(options["cfg"]),
            "--port", str(port)],
            env=script_env(offline=False))
