"""Higgs TTS 3 through its transformers port, for hosts without SGLang-Omni.

The same weights as `higgs.py`, run in-process instead of behind an SGLang
worker. On Apple silicon it is the only way Higgs runs at all. It serves the
same speech contract, so a client cannot tell the two apart except by speed.
"""
from __future__ import annotations

from pathlib import Path

from ..types import Format, ModelSet, Task
from .base import LaunchPlan, SPEECH_PATHS
from .probe import http_ok


class HiggsLocalEngine:
    id = "higgs_local"
    display_name = "Higgs TTS (transformers)"

    def __init__(self, binary: str | None = None, server: str | None = None,
                 model_options: dict[str, dict] | None = None,
                 max_batch: int = 1, batch_window_ms: int = 100) -> None:
        if type(max_batch) is not int or not 1 <= max_batch <= 16:
            raise ValueError("Higgs max_batch must be an integer from 1 to 16")
        if type(batch_window_ms) is not int or not 0 <= batch_window_ms <= 2000:
            raise ValueError("Higgs batch_window_ms must be an integer from 0 to 2000")
        self.max_batch = max_batch
        self.batch_window_ms = batch_window_ms
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
        return {"instruction_required": False, "instruction_visible": False,
                "language_visible": False, "speaker_visible": False,
                "seed_supported": True, "reference_supported": True}

    def plan(self, model: ModelSet, port: int, params: dict) -> LaunchPlan:
        if not self.supports(model):
            raise ValueError(f"Higgs (transformers) is not configured for {model.name}")
        if params:
            raise ValueError("Higgs has no instance settings")
        checkpoint = Path(model.entrypoint)
        model_dir = checkpoint.parent if checkpoint.is_file() else checkpoint
        return LaunchPlan(argv=[
            self.binary, self.server, "--backend", "higgs",
            "--model-path", str(model_dir), "--port", str(port),
            "--max-batch", str(self.max_batch),
            "--batch-window-ms", str(self.batch_window_ms)],
            env={"PYTHONUNBUFFERED": "1", "HF_HUB_OFFLINE": "1",
                 "PYTORCH_ENABLE_MPS_FALLBACK": "1",
                 "PYTHONPATH": str(Path(__file__).resolve().parents[2])})

    def ready(self, port: int) -> bool:
        return http_ok(port)

    def concurrency(self, params: dict) -> int:
        # One model copy; parallel requests are decoded together in one batch.
        return self.max_batch

    def needs_mb(self, model: ModelSet, params: dict,
                 card_total_mb: float) -> float:
        return float(self.model_options[model.name]["memory_reservation_mb"])

    def api_paths(self, task: Task = Task.SPEECH_SYNTHESIS) -> tuple[str, ...]:
        return SPEECH_PATHS if task is Task.SPEECH_SYNTHESIS else ()
