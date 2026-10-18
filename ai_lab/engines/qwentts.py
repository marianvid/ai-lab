"""Qwen3-TTS runtime for VoiceDesign and CustomVoice checkpoints."""
from __future__ import annotations

from ..types import ModelSet, Task
from .base import SPEECH_PATHS, LaunchPlan
from .isolated import ConfiguredEngine, model_directory, script_env, ui_port, with_headroom_mb

# The two kinds of Qwen3-TTS checkpoint this runtime serves.
MODES = frozenset({"voice-design", "custom-voice"})
# Activations, attention buffers and CUDA allocations beside the weights.
HEADROOM_MB = 2048


class QwenTtsEngine(ConfiguredEngine):
    """Qwen3-TTS, with each checkpoint's mode (`model_modes`) from the configuration."""

    id = "qwentts"
    display_name = "Qwen3-TTS"
    task = Task.SPEECH_SYNTHESIS
    paths = SPEECH_PATHS
    script = ("speech", "server.py")

    def __init__(self, binary: str | None = None, server: str | None = None,
                 model_modes: dict[str, str] | None = None) -> None:
        # Configured as `model_modes` (name -> mode), the one option it has.
        super().__init__(binary, server, {name: {"mode": mode}
                                          for name, mode in (model_modes or {}).items()})
        self.model_modes = dict(model_modes or {})

    def configured(self, model_name: str) -> bool:
        """Whether this checkpoint is configured as one of the two modes."""
        return self.model_modes.get(model_name) in MODES

    def speech_form(self, model_name: str) -> dict:
        """What the speech page shows: an instruction (VoiceDesign) or a speaker (CustomVoice)."""
        mode = self.model_modes.get(model_name)
        return {"instruction_required": mode == "voice-design",
                "instruction_visible": True,
                "speaker_visible": mode == "custom-voice",
                "language_visible": True,
                "speaker_label": "Speaker",
                "seed_supported": True, "reference_supported": False}

    def plan(self, model: ModelSet, port: int, params: dict) -> LaunchPlan:
        """The speech host with the Qwen backend in this checkpoint's mode."""
        self._options(model, params)
        return LaunchPlan(argv=[self.binary, self.server, "--backend", "qwen",
                                "--model-path", model_directory(model),
                                "--mode", self.model_modes[model.name],
                                "--ui-port", ui_port(port),
                                "--port", str(port)],
                          env=script_env(offline=False))

    def needs_mb(self, model: ModelSet, params: dict, card_total_mb: float) -> float:
        """The weights with room for the runtime's working memory."""
        return with_headroom_mb(model, HEADROOM_MB)
