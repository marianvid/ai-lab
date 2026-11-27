"""ONNX Runtime for small speech models that do not need the GPU."""

from __future__ import annotations

from ...types import Format, ModelSet, Task
from ..base import NEEDS_NOTHING_MB, VAD_PATHS
from ..isolated import IsolatedEngine


class OnnxEngine(IsolatedEngine):
    """Silero voice activity detection on the processor."""

    missing_reason = "Not installed"   # said on the Engines page when it is missing
    id = "onnx"
    display_name = "ONNX Runtime"
    task = Task.VAD
    paths = VAD_PATHS
    script = ("audio", "server.py")
    served_formats = frozenset({Format.ONNX})
    finds_python = True
    backend = "silero"

    def needs_mb(self, model: ModelSet, params: dict, card_total_mb: float) -> float:
        """Nothing of the card: it runs on the processor (`NEEDS_NOTHING_MB`, not 0)."""
        return NEEDS_NOTHING_MB
