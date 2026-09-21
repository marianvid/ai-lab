"""VoxCPM speech engine; checkpoint and inference settings come from config."""
from __future__ import annotations

from pathlib import Path

from ..types import Format, ModelSet, Task
from .base import LaunchPlan, SPEECH_PATHS
from .probe import http_ok


class VoxCpmEngine:
    id = "voxcpm"
    display_name = "VoxCPM"

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
        return {"instruction_required": False, "instruction_visible": True,
                "speaker_visible": False, "language_visible": False}

    def plan(self, model: ModelSet, port: int, params: dict) -> LaunchPlan:
        if not self.supports(model):
            raise ValueError(f"VoxCPM is not configured for {model.name}")
        if params:
            raise ValueError("VoxCPM has no instance settings")
        options = self.model_options[model.name]
        checkpoint = Path(model.entrypoint)
        model_dir = checkpoint.parent if checkpoint.is_file() else checkpoint
        return LaunchPlan(argv=[
            self.binary, self.server, "--backend", "voxcpm",
            "--model-path", str(model_dir),
            "--cfg-value", str(options.get("cfg_value", 2.0)),
            "--inference-timesteps", str(options.get("inference_timesteps", 10)),
            "--port", str(port), "--ui-port", str(port + 10000)],
            env={"PYTHONUNBUFFERED": "1", "HF_HUB_OFFLINE": "1",
                 "PYTHONPATH": str(Path(__file__).resolve().parents[2])})

    def ready(self, port: int) -> bool:
        return http_ok(port)

    def concurrency(self, params: dict) -> int:
        return 1

    def needs_mb(self, model: ModelSet, params: dict,
                 card_total_mb: float) -> float:
        weights_mb = model.size_bytes / (1024 * 1024)
        return max(weights_mb * 1.5, weights_mb + 2048)

    def api_paths(self, task: Task = Task.SPEECH_SYNTHESIS) -> tuple[str, ...]:
        return SPEECH_PATHS if task is Task.SPEECH_SYNTHESIS else ()
