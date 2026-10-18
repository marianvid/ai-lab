"""ONNX Runtime for small speech models that do not need the GPU."""

from __future__ import annotations

from ..types import Format, ModelSet, Task
from .base import VAD_PATHS
from .isolated import IsolatedEngine


class OnnxEngine(IsolatedEngine):
    """Silero voice activity detection on the processor."""

    id = "onnx"
    display_name = "ONNX Runtime"
    task = Task.VAD
    paths = VAD_PATHS
    script = ("audio", "server.py")
    served_formats = frozenset({Format.ONNX})
    finds_python = True
    backend = "silero"

    def needs_mb(self, model: ModelSet, params: dict, card_total_mb: float) -> float:
        """Nothing: it runs on the processor."""
        return 0.0
