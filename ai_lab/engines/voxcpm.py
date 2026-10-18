"""VoxCPM speech engine; checkpoint and inference settings come from config."""
from __future__ import annotations

from ..types import ModelSet, Task
from .base import SPEECH_PATHS, LaunchPlan
from .isolated import ConfiguredEngine, model_directory, script_env, ui_port, with_headroom_mb

# Defaults for the two inference settings a model's options may set.
CFG_VALUE = 2.0
INFERENCE_TIMESTEPS = 10
HEADROOM_MB = 2048


class VoxCpmEngine(ConfiguredEngine):
    """VoxCPM, with the upstream editor beside the API."""

    id = "voxcpm"
    display_name = "VoxCPM"
    task = Task.SPEECH_SYNTHESIS
    paths = SPEECH_PATHS
    script = ("speech", "server.py")

    def speech_form(self, model_name: str) -> dict:
        """What the speech page shows: an instruction, a seed and a reference clip."""
        return {"instruction_required": False, "instruction_visible": True,
                "speaker_visible": False, "language_visible": False,
                "seed_supported": True, "reference_supported": True}

    def plan(self, model: ModelSet, port: int, params: dict) -> LaunchPlan:
        """The speech host with the VoxCPM backend and this model's settings."""
        options = self._options(model, params)
        return LaunchPlan(argv=[
            self.binary, self.server, "--backend", "voxcpm",
            "--model-path", model_directory(model),
            "--cfg-value", str(options.get("cfg_value", CFG_VALUE)),
            "--inference-timesteps",
            str(options.get("inference_timesteps", INFERENCE_TIMESTEPS)),
            "--port", str(port), "--ui-port", ui_port(port)],
            env=script_env())

    def needs_mb(self, model: ModelSet, params: dict, card_total_mb: float) -> float:
        """The weights with room for the runtime's working memory."""
        return with_headroom_mb(model, HEADROOM_MB)
