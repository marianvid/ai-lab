"""NVIDIA NeMo Speech models, served behind AI-Lab's common audio API."""

from __future__ import annotations

from ...network import ALL_INTERFACES
from ...types import Format, ModelSet, Task
from ..base import DIARIZATION_PATHS, TRANSCRIPTION_PATHS, LaunchPlan, ParamSpec, validate
from ..isolated import IsolatedEngine, weights_mb

PARAMS = (
    ParamSpec("precision", "Precision", "choice", "bf16",
              choices=("bf16", "fp16", "fp32"), group="memory",
              help="Arithmetic used by the speech model. BF16 is the native "
                   "choice on this Blackwell card; FP32 uses roughly twice "
                   "the memory and is mainly a diagnostic setting."),
)
# Which request shapes each task answers.
_PATHS = {Task.TRANSCRIPTION: TRANSCRIPTION_PATHS, Task.DIARIZATION: DIARIZATION_PATHS}


class NemoEngine(IsolatedEngine):
    """NeMo transcription (Canary, Parakeet, Nemotron) and diarization (Sortformer)."""

    id = "nemo"
    display_name = "NVIDIA NeMo Speech"
    task = Task.TRANSCRIPTION
    script = ("audio", "server.py")
    served_formats = frozenset({Format.NEMO})
    settings = PARAMS
    finds_python = True

    def tasks(self) -> frozenset[Task]:
        """Transcription and diarization."""
        return frozenset(_PATHS)

    def plan(self, model: ModelSet, port: int, params: dict) -> LaunchPlan:
        """The audio server with the NeMo or Sortformer backend, at the chosen precision."""
        if model.format is not Format.NEMO:
            raise ValueError(f"NeMo cannot load {model.format.value} models")
        if model.task not in self.tasks():
            raise ValueError(f"NeMo cannot perform {model.task.value}")
        if not model.complete:
            raise ValueError(f"{model.name} is incomplete")
        settings = validate(PARAMS, params)
        return LaunchPlan(
            argv=[self.binary, self.server,
                  "--backend", ("sortformer" if model.task is Task.DIARIZATION
                                else "nemo"),
                  "--model", model.entrypoint,
                  "--name", model.name,
                  "--host", ALL_INTERFACES,
                  "--port", str(port),
                  "--precision", str(settings["precision"])],
            env={"PYTHONUNBUFFERED": "1"},
        )

    def needs_mb(self, model: ModelSet, params: dict, card_total_mb: float) -> float:
        """The weights, twice over at FP32."""
        multiplier = 2.0 if validate(PARAMS, params)["precision"] == "fp32" else 1.0
        return weights_mb(model) * multiplier

    def api_paths(self, task: Task | None = None) -> tuple[str, ...]:
        """Transcription or diarization shapes, by task."""
        return _PATHS.get(task or self.task, ())
