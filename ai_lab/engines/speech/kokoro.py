"""Kokoro speech engine with checkpoint and voice settings from configuration."""
from __future__ import annotations

from ...types import ModelSet, Task
from ..base import SPEECH_PATHS, LaunchPlan
from ..isolated import ConfiguredEngine, script_env, ui_port

# What each model's options must name.
REQUIRED_OPTIONS = ("language_code", "default_voice", "repo_id")


class KokoroEngine(ConfiguredEngine):
    """Kokoro, with its language, default voice and voice repository per model."""

    id = "kokoro"
    display_name = "Kokoro"
    task = Task.SPEECH_SYNTHESIS
    paths = SPEECH_PATHS
    script = ("speech", "server.py")

    def speech_form(self, model_name: str) -> dict:
        """What the speech page shows: a voice, with the configured default as the hint."""
        options = self.model_options.get(model_name, {})
        return {"instruction_required": False,
                "instruction_visible": False,
                "speaker_visible": True,
                "language_visible": False,
                "speaker_label": "Voice",
                "speaker_hint": "Default: " + options.get("default_voice", ""),
                "seed_supported": False, "reference_supported": False}

    def plan(self, model: ModelSet, port: int, params: dict) -> LaunchPlan:
        """The speech host with the Kokoro backend and its own editor."""
        options = self._options(model, params)
        if any(not options.get(key) for key in REQUIRED_OPTIONS):
            raise ValueError(f"Kokoro settings are incomplete for {model.name}")
        return LaunchPlan(argv=[
            self.binary, self.server, "--backend", "kokoro",
            "--model-path", model.entrypoint,
            "--language-code", options["language_code"],
            "--default-voice", options["default_voice"],
            "--repo-id", options["repo_id"], "--port", str(port),
            "--ui-port", ui_port(port)],
            env=script_env())
