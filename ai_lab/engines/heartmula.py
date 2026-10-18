"""HeartMuLa lyric-conditioned music from a configurable local checkpoint."""
from __future__ import annotations

from ..types import ModelSet, Task
from .base import MUSIC_PATHS, LaunchPlan
from .isolated import ConfiguredEngine, model_directory, script_env, ui_port


class HeartMulaEngine(ConfiguredEngine):
    """HeartMuLa songs from lyrics, with the HeartMuse page beside the API."""

    id = "heartmula"
    display_name = "HeartMuLa"
    task = Task.MUSIC_GENERATION
    paths = MUSIC_PATHS
    script = ("music", "heartmula_server.py")
    reserves_memory = True
    location_settings = ("bundle_root", "heartmuse_root", "output_root")

    def music_form(self, model_name: str) -> dict:
        """What the music page asks for: lyrics are required."""
        return {"lyrics_required": True}

    def plan(self, model: ModelSet, port: int, params: dict) -> LaunchPlan:
        """The HeartMuLa server for this checkpoint, with HeartMuse on the page port."""
        options = self._options(
            model, params, "HeartMuLa bundle, output and HeartMuse paths must be configured")
        where = self.locations
        return LaunchPlan(argv=[
            self.binary, self.server, "--bundle-root", where["bundle_root"],
            "--checkpoint-path", model_directory(model),
            "--checkpoint-subdir", options["checkpoint_subdir"],
            "--output-root", where["output_root"],
            "--version", options["version"],
            "--topk", str(options["topk"]),
            "--temperature", str(options["temperature"]),
            "--cfg-scale", str(options["cfg_scale"]),
            "--port", str(port), "--ui-port", ui_port(port),
            "--heartmuse-root", where["heartmuse_root"]],
            env=script_env())
