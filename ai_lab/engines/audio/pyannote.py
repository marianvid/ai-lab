"""Local pyannote.audio speaker diarization pipelines."""

from __future__ import annotations

from ...types import Format, Task
from ..base import DIARIZATION_PATHS
from ..isolated import IsolatedEngine

PARAMS = ()


class PyannoteEngine(IsolatedEngine):
    """pyannote.audio pipelines: who speaks when."""

    id = "pyannote"
    display_name = "pyannote.audio"
    task = Task.DIARIZATION
    paths = DIARIZATION_PATHS
    script = ("audio", "server.py")
    served_formats = frozenset({Format.PYANNOTE})
    settings = PARAMS
    finds_python = True
    backend = "pyannote"
