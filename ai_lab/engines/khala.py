"""Khala music via the isolated Apple Silicon vanilla generator."""
from __future__ import annotations

from ..types import ModelSet, Task
from .base import MUSIC_PATHS, LaunchPlan
from .isolated import ConfiguredEngine, model_directory, script_env, ui_port, with_headroom_mb

# Khala's song lengths are buckets, not seconds: these are the defaults.
DEFAULT_BUCKET = 0
MAXIMUM_BUCKET = 4
# What the generator holds beside its weights, at least.
HEADROOM_MB = 2048


class KhalaEngine(ConfiguredEngine):
    """Khala songs, run by its own generator script."""

    id = "khala"
    display_name = "Khala"
    task = Task.MUSIC_GENERATION
    paths = MUSIC_PATHS
    script = ("music", "khala_server.py")
    location_settings = ("generator_script", "output_root")

    def music_form(self, model_name: str) -> dict:
        """What the music page shows: a length bucket, with this model's range."""
        options = self.model_options.get(model_name, {})
        return {"duration_kind": "bucket",
                "default_bucket": options.get("default_bucket", DEFAULT_BUCKET),
                "maximum_bucket": options.get("maximum_bucket", MAXIMUM_BUCKET)}

    def plan(self, model: ModelSet, port: int, params: dict) -> LaunchPlan:
        """The Khala host for this checkpoint."""
        options = self._options(
            model, params, "Khala generator and output path must be configured")
        return LaunchPlan(argv=[
            self.binary, self.server,
            "--generator-script", self.locations["generator_script"],
            "--model-path", model_directory(model),
            "--output-root", self.locations["output_root"],
            "--default-bucket", str(options.get("default_bucket", DEFAULT_BUCKET)),
            "--maximum-bucket", str(options.get("maximum_bucket", MAXIMUM_BUCKET)),
            "--port", str(port), "--ui-port", ui_port(port)],
            env=script_env())

    def needs_mb(self, model: ModelSet, params: dict, card_total_mb: float) -> float:
        """The weights with room for the generator's working memory."""
        return with_headroom_mb(model, HEADROOM_MB)
