"""Stable Audio 3: instrumental music and sound from a text prompt."""
from __future__ import annotations

from ..types import ModelSet, Task
from .base import MUSIC_PATHS, LaunchPlan
from .isolated import ConfiguredEngine, model_directory, script_env, ui_port


class StableAudio3Engine(ConfiguredEngine):
    """Stable Audio 3, with Stability's own Gradio page beside the API."""

    id = "stableaudio3"
    display_name = "Stable Audio 3"
    task = Task.MUSIC_GENERATION
    paths = MUSIC_PATHS
    script = ("music", "stableaudio3_server.py")
    reserves_memory = True

    def music_form(self, model_name: str) -> dict:
        """What the music page shows: a length in seconds, instrumental only."""
        return {"lyrics_required": False, "duration_kind": "seconds",
                "instrumental_only": True}

    def plan(self, model: ModelSet, port: int, params: dict) -> LaunchPlan:
        """The Stable Audio 3 host with this model's sampling settings."""
        options = self._options(model, params)
        return LaunchPlan(argv=[
            self.binary, self.server, "--model-path", model_directory(model),
            "--steps", str(options["steps"]),
            "--cfg-scale", str(options["cfg_scale"]), "--port", str(port),
            "--ui-port", ui_port(port)],
            # On a Mac, let an operation Metal lacks run on the CPU instead of
            # failing the song. Ignored on Linux.
            env=script_env(GRADIO_ANALYTICS_ENABLED="False",
                           PYTORCH_ENABLE_MPS_FALLBACK="1"),
            web_ui=True)

    def web_ui(self) -> str:
        """Stability's official Gradio page, on the instance port + 10000."""
        return "native"
