"""Kokoro speech engine with checkpoint and voice settings from configuration."""
from __future__ import annotations

from pathlib import Path

from ..types import Format, ModelSet, Task
from .base import LaunchPlan, SPEECH_PATHS
from .probe import http_ok


class KokoroEngine:
    id = "kokoro"
    display_name = "Kokoro"

    def __init__(self, binary: str | None = None, server: str | None = None,
                 model_options: dict[str, dict] | None = None) -> None:
        self.binary = binary or "python"
        self.server = server or str(Path(__file__).resolve().parents[1] /
                                    "speech" / "server.py")
        self.model_options = dict(model_options or {})

    def formats(self) -> frozenset[Format]:
        return frozenset({Format.SAFETENSORS})

    def tasks(self) -> frozenset[Task]:
        return frozenset({Task.SPEECH_SYNTHESIS})

    def params(self, task: Task = Task.SPEECH_SYNTHESIS):
        return ()

    def supports(self, model: ModelSet) -> bool:
        return (model.format is Format.SAFETENSORS
                and model.task is Task.SPEECH_SYNTHESIS
                and model.name in self.model_options)

    def speech_form(self, model_name: str) -> dict:
        options = self.model_options.get(model_name, {})
        return {"instruction_required": False,
                "instruction_visible": False,
                "speaker_visible": True,
                "language_visible": False,
                "speaker_label": "Voice",
                "speaker_hint": "Default: " + options.get("default_voice", "")}

    def plan(self, model: ModelSet, port: int, params: dict) -> LaunchPlan:
        if not self.supports(model):
            raise ValueError(f"Kokoro is not configured for {model.name}")
        if params:
            raise ValueError("Kokoro has no instance settings")
        options = self.model_options[model.name]
        required = ("language_code", "default_voice", "repo_id")
        if any(not options.get(key) for key in required):
            raise ValueError(f"Kokoro settings are incomplete for {model.name}")
        return LaunchPlan(argv=[
            self.binary, self.server, "--backend", "kokoro",
            "--model-path", model.entrypoint,
            "--language-code", options["language_code"],
            "--default-voice", options["default_voice"],
            "--repo-id", options["repo_id"], "--port", str(port),
            "--ui-port", str(port + 10000)],
            env={"PYTHONUNBUFFERED": "1", "HF_HUB_OFFLINE": "1",
                 "PYTHONPATH": str(Path(__file__).resolve().parents[2])})

    def ready(self, port: int) -> bool:
        return http_ok(port)

    def concurrency(self, params: dict) -> int:
        return 1

    def needs_mb(self, model: ModelSet, params: dict,
                 card_total_mb: float) -> float:
        return model.size_bytes / (1024 * 1024)

    def api_paths(self, task: Task = Task.SPEECH_SYNTHESIS) -> tuple[str, ...]:
        return SPEECH_PATHS if task is Task.SPEECH_SYNTHESIS else ()
