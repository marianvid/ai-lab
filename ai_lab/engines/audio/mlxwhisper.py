"""Whisper transcription accelerated by MLX on Apple silicon."""

from __future__ import annotations

from ...types import Task
from ..base import TRANSCRIPTION_PATHS
from ..isolated import IsolatedEngine


class MlxWhisperEngine(IsolatedEngine):
    """Whisper checkpoints in MLX format, behind the shared audio server."""

    id = "mlxwhisper"
    display_name = "MLX Whisper"
    task = Task.TRANSCRIPTION
    paths = TRANSCRIPTION_PATHS
    script = ("audio", "server.py")
    finds_python = True
    backend = "mlx-whisper"
