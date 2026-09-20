"""Higgs TTS via a supervised SGLang-Omni worker."""
from __future__ import annotations

from pathlib import Path

from ..types import Format, ModelSet, Task
from .base import LaunchPlan, SPEECH_PATHS
from .probe import http_ok


class HiggsEngine:
    id = "higgs"
    display_name = "Higgs TTS"

    def __init__(self, binary: str | None = None, server: str | None = None,
                 worker_binary: str | None = None,
                 model_options: dict[str, dict] | None = None) -> None:
        self.binary = binary or "python"
        self.server = server or str(Path(__file__).resolve().parents[1] /
                                    "speech" / "higgs_server.py")
        self.worker_binary = worker_binary or "sgl-omni"
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
        return {"instruction_required": False, "instruction_visible": False,
                "language_visible": False, "speaker_visible": True,
                "speaker_label": "Voice", "speaker_hint": "Default voice"}

    def plan(self, model: ModelSet, port: int, params: dict) -> LaunchPlan:
        if not self.supports(model):
            raise ValueError(f"Higgs is not configured for {model.name}")
        if params:
            raise ValueError("Higgs has no instance settings")
        options = self.model_options[model.name]
        checkpoint = Path(model.entrypoint)
        model_dir = checkpoint.parent if checkpoint.is_file() else checkpoint
        return LaunchPlan(argv=[
            self.binary, self.server,
            "--worker-binary", self.worker_binary,
            "--model-path", str(model_dir),
            "--worker-port", str(options["worker_port"]),
            "--mem-fraction-static", str(options["mem_fraction_static"]),
            "--port", str(port)],
            env={"PYTHONUNBUFFERED": "1", "HF_HUB_OFFLINE": "1",
                 "PYTHONPATH": str(Path(__file__).resolve().parents[2])})

    def ready(self, port: int) -> bool:
        return http_ok(port)

    def concurrency(self, params: dict) -> int:
        return 1

    def needs_mb(self, model: ModelSet, params: dict,
                 card_total_mb: float) -> float:
        return float(self.model_options[model.name]["memory_reservation_mb"])

    def api_paths(self, task: Task = Task.SPEECH_SYNTHESIS) -> tuple[str, ...]:
        return SPEECH_PATHS if task is Task.SPEECH_SYNTHESIS else ()
