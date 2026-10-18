"""Higgs TTS 3 through its transformers port, for hosts without SGLang-Omni.

The same weights as `higgs.py`, run in-process instead of behind an SGLang
worker. On Apple silicon it is the only way Higgs runs at all. It serves the
same speech contract, so a client cannot tell the two apart except by speed.
"""
from __future__ import annotations

from ...types import ModelSet, Task
from ..base import SPEECH_PATHS, LaunchPlan
from ..isolated import ConfiguredEngine, model_directory, script_env, ui_port

# Limits on decoding several requests together.
MAX_BATCH = 16
MAX_BATCH_WINDOW_MS = 2000


class HiggsLocalEngine(ConfiguredEngine):
    """Higgs TTS 3 in the speech host's own process, batching requests that arrive together."""

    id = "higgs_local"
    display_name = "Higgs TTS (transformers)"
    task = Task.SPEECH_SYNTHESIS
    paths = SPEECH_PATHS
    script = ("speech", "server.py")
    reserves_memory = True

    def __init__(self, binary: str | None = None, server: str | None = None,
                 model_options: dict[str, dict] | None = None,
                 max_batch: int = 1, batch_window_ms: int = 100) -> None:
        if type(max_batch) is not int or not 1 <= max_batch <= MAX_BATCH:
            raise ValueError(f"Higgs max_batch must be an integer from 1 to {MAX_BATCH}")
        if type(batch_window_ms) is not int or not 0 <= batch_window_ms <= MAX_BATCH_WINDOW_MS:
            raise ValueError("Higgs batch_window_ms must be an integer from 0 to "
                             f"{MAX_BATCH_WINDOW_MS}")
        super().__init__(binary, server, model_options)
        self.max_batch = max_batch
        self.batch_window_ms = batch_window_ms

    def speech_form(self, model_name: str) -> dict:
        """What the speech page shows: a seed and a reference clip, no voice list."""
        return {"instruction_required": False, "instruction_visible": False,
                "language_visible": False, "speaker_visible": False,
                "seed_supported": True, "reference_supported": True}

    def plan(self, model: ModelSet, port: int, params: dict) -> LaunchPlan:
        """The speech host with the Higgs backend and the upstream playground page."""
        self._options(model, params)
        return LaunchPlan(argv=[
            self.binary, self.server, "--backend", "higgs",
            "--model-path", model_directory(model), "--port", str(port),
            "--max-batch", str(self.max_batch),
            "--batch-window-ms", str(self.batch_window_ms),
            # The upstream Higgs playground page, for direct use in a browser.
            "--ui-port", ui_port(port)],
            env=script_env(PYTORCH_ENABLE_MPS_FALLBACK="1"))

    def concurrency(self, params: dict) -> int:
        """One model copy; parallel requests are decoded together in one batch."""
        return self.max_batch
