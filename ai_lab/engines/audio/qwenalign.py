"""Qwen forced alignment with local weights and the shared audio gateway."""
from __future__ import annotations

from ...network import LOOPBACK
from ...types import ModelSet, Task
from ..base import ALIGNMENT_PATHS, LaunchPlan
from ..isolated import IsolatedEngine, model_directory, script_env, with_headroom_mb

# What the aligner holds beside its weights, at least.
HEADROOM_MB = 1024


class QwenAlignEngine(IsolatedEngine):
    """Qwen3 forced aligner: when each word of a transcript is spoken."""

    id = "qwenalign"
    display_name = "Qwen Forced Aligner"
    task = Task.ALIGNMENT
    paths = ALIGNMENT_PATHS
    script = ("audio", "server.py")

    def plan(self, model: ModelSet, port: int, params: dict) -> LaunchPlan:
        """The audio server with the qwen-align backend."""
        self._checked(model, params)
        return LaunchPlan(argv=[
            self.binary, self.server, "--backend", "qwen-align",
            "--model", model_directory(model), "--name", model.name,
            "--host", LOOPBACK, "--port", str(port)],
            env=script_env())

    def needs_mb(self, model: ModelSet, params: dict, card_total_mb: float) -> float:
        """The weights with room for the aligner's working memory."""
        return with_headroom_mb(model, HEADROOM_MB)
