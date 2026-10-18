"""MuLaCover source-audio arrangement engine."""
from __future__ import annotations

from ...types import ModelSet, Task
from ..base import MUSIC_PATHS, LaunchPlan
from ..isolated import ConfiguredEngine, model_directory, script_env
from . import music_server


class MulaCoverEngine(ConfiguredEngine):
    """MuLaCover covers, from a reference recording and lyrics."""

    id = "mulacover"
    display_name = "MuLaCover"
    task = Task.MUSIC_GENERATION
    location_settings = ("bundle_root", "output_root")
    reserves_memory = True
    paths = MUSIC_PATHS
    script = music_server("mulacover")

    def music_form(self, model_name: str) -> dict:
        """What the music page asks for: a reference recording and lyrics, no length."""
        return {"reference_audio_required": True, "lyrics_required": True,
                "duration_kind": "none"}

    def plan(self, model: ModelSet, port: int, params: dict) -> LaunchPlan:
        """The MuLaCover host for this checkpoint."""
        options = self._options(
            model, params, "MuLaCover bundle and output paths must be configured")
        return LaunchPlan(argv=[
            self.binary, self.server, "--bundle-root", self.locations["bundle_root"],
            "--checkpoint-path", model_directory(model),
            "--checkpoint-subdir", options["checkpoint_subdir"],
            "--output-root", self.locations["output_root"],
            "--topk", str(options["topk"]),
            "--temperature", str(options["temperature"]),
            "--cfg-scale", str(options["cfg_scale"]),
            "--port", str(port)],
            env=script_env())
