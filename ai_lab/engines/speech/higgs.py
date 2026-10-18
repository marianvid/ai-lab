"""Higgs TTS via a supervised SGLang-Omni worker."""
from __future__ import annotations

from ...types import ModelSet, Task
from ..base import SPEECH_PATHS, LaunchPlan
from ..isolated import ConfiguredEngine, model_directory, script_env, ui_port

# How many speech requests the SGLang worker may be sent at once, at most.
MAX_PARALLEL = 64


class HiggsEngine(ConfiguredEngine):
    """Higgs TTS 3 in an SGLang-Omni worker, behind AI-Lab's speech contract."""

    id = "higgs"
    display_name = "Higgs TTS"
    task = Task.SPEECH_SYNTHESIS
    paths = SPEECH_PATHS
    script = ("speech", "higgs_server.py")
    reserves_memory = True

    def __init__(self, binary: str | None = None, server: str | None = None,
                 worker_binary: str | None = None,
                 model_options: dict[str, dict] | None = None,
                 max_parallel: int = 1) -> None:
        super().__init__(binary, server, model_options)
        self.worker_binary = worker_binary or "sgl-omni"
        # How many speech requests the SGLang worker is sent at once. The
        # worker batches them itself; the limit exists because the stages
        # outside its static memory share (audio encoder, vocoder) grow with
        # every request in flight. Measured on the 32 GB card at 0.8: eight
        # succeed together, sixteen ran out of memory.
        if type(max_parallel) is not int or not 1 <= max_parallel <= MAX_PARALLEL:
            raise ValueError(f"Higgs max_parallel must be an integer from 1 to {MAX_PARALLEL}")
        self.max_parallel = max_parallel

    def speech_form(self, model_name: str) -> dict:
        """What the speech page shows: a voice, a seed, a reference clip."""
        return {"instruction_required": False, "instruction_visible": False,
                "language_visible": False, "speaker_visible": True,
                "speaker_label": "Voice", "speaker_hint": "Default voice",
                "seed_supported": True, "reference_supported": True}

    def plan(self, model: ModelSet, port: int, params: dict) -> LaunchPlan:
        """The host that starts the worker and the playground page."""
        options = self._options(model, params)
        return LaunchPlan(argv=[
            self.binary, self.server,
            "--worker-binary", self.worker_binary,
            "--model-path", model_directory(model),
            "--worker-port", str(options["worker_port"]),
            "--mem-fraction-static", str(options["mem_fraction_static"]),
            "--max-parallel", str(self.max_parallel),
            "--port", str(port), "--ui-port", ui_port(port)],
            env=script_env())

    def concurrency(self, params: dict) -> int:
        """As many as the worker is sent at once."""
        return self.max_parallel
