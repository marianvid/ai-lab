"""Local PaddleOCR/PaddleX text-recognition pipelines."""

from __future__ import annotations

from ...types import Format, Task
from ..base import OCR_PATHS
from ..isolated import IsolatedEngine

PARAMS = ()


class PaddleOcrEngine(IsolatedEngine):
    """PaddleOCR detection and recognition models, behind the image server."""

    id = "paddleocr"
    display_name = "PaddleOCR"
    task = Task.OCR
    paths = OCR_PATHS
    script = ("images", "server.py")
    served_formats = frozenset({Format.PADDLEOCR})
    settings = PARAMS
    finds_python = True
    backend = "paddleocr"
