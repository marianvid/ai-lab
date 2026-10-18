#!/usr/bin/env python3
"""HTTP adapter for a local PaddleOCR/PaddleX pipeline.

Executable by a runtime-specific Python environment, the same way
`ai_lab/audio/server.py` is: PaddleOCR stays isolated from the manager, and
only this small HTTP contract is shared with it (`engine_host.http_host`'s
`UploadHost`).

The response is AI-Lab's own stable OCR schema, not PaddleOCR's native
output shape — every client depends on this shape, not on
whatever PaddleOCR happens to return this version.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from threading import Lock

# This file is launched directly by the runtime's own Python environment.
# Give it access to the small shared modules beside this script.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ai_lab.engine_host.http_host import Form, UploadHost, form_text, serve_uploads, uploaded_file
from ai_lab.network import LOOPBACK


class PaddleOcrBackend:
    """Text in an image, with PaddleOCR's detection and recognition models."""

    owner = "paddleocr"
    path = "/v1/images/ocr"

    def __init__(self, model_path: str, precision: str = "fp32") -> None:
        from paddleocr import PaddleOCR

        root = Path(model_path)
        detection, recognition, classification = self.model_directories(root)
        kwargs = {
            "text_detection_model_dir": str(detection),
            "text_detection_model_name": detection.name,
            "text_recognition_model_dir": str(recognition),
            "text_recognition_model_name": recognition.name,
            "use_doc_orientation_classify": classification is not None,
            "use_doc_unwarping": False,
            "use_textline_orientation": classification is not None,
        }
        self.pipeline = PaddleOCR(**kwargs)
        self.lock = Lock()

    @staticmethod
    def model_directories(root: Path) -> tuple[Path, Path, Path | None]:
        """Resolve a managed OCR pair without downloading hidden defaults.

        A bundled model may contain ``det/`` and ``rec/``. Official PaddleOCR
        Hugging Face releases are separate ``*_det`` and ``*_rec`` packages;
        AI-Lab stores those as siblings and the detection package is the model
        selected by the instance. Both forms remain explicit and auditable.
        """
        detection, recognition = root / "det", root / "rec"
        classification = root / "cls"
        if detection.is_dir() and recognition.is_dir():
            return detection, recognition, classification if classification.is_dir() else None
        if root.name.endswith("_det"):
            sibling = root.with_name(root.name[:-4] + "_rec")
            if sibling.is_dir():
                return root, sibling, None
        raise ValueError(
            "OCR model requires det/ and rec/ directories, or managed sibling "
            "packages whose names end in _det and _rec")

    def recognize(self, image_path: str, language: str | None = None) -> dict:
        """The image's text, line by line with confidence and outline, in AI-Lab's schema."""
        with self.lock:
            result = list(self.pipeline.predict(image_path))
        lines: list[dict] = []
        orientation = 0
        for page in result:
            payload = _page_payload(page)
            if payload is None:
                continue
            lines += _page_lines(payload)
            orientation = int(payload.get("doc_preprocessor_res", {}).get(
                "angle", orientation) or orientation)
        text = "\n".join(str(line["text"]) for line in lines)
        return {
            "text": text,
            "lines": lines,
            "language": language or "",
            "orientation_degrees": orientation,
            "warnings": [] if lines else ["no text recognized"],
        }


def _page_payload(page) -> dict | None:
    """One page of PaddleOCR's output as a dictionary, whichever form it came in."""
    payload = getattr(page, "json", page)
    if callable(payload):
        payload = payload()
    if isinstance(payload, str):
        payload = json.loads(payload)
    if isinstance(payload, dict) and isinstance(payload.get("res"), dict):
        payload = payload["res"]
    return payload if isinstance(payload, dict) else None


def _page_lines(payload: dict) -> list[dict]:
    """The recognised lines of one page: text, confidence and polygon."""
    texts = payload.get("rec_texts", [])
    scores = payload.get("rec_scores", [])
    polygons = payload.get("rec_polys") or payload.get("dt_polys") or []
    lines = []
    for index, text in enumerate(texts):
        polygon = polygons[index] if index < len(polygons) else []
        score = scores[index] if index < len(scores) else 0.0
        lines.append({"text": str(text), "confidence": float(score),
                      "polygon": [[float(x), float(y)] for x, y in polygon]})
    return lines


class OcrHost(UploadHost):
    """Answers an uploaded image with the text PaddleOCR reads in it."""

    backend: PaddleOcrBackend

    def answer(self, fields: Form) -> dict:
        """The image, saved to a temporary file and read."""
        filename, image = uploaded_file(fields, "image")
        suffix = Path(filename or "image.png").suffix or ".png"
        with tempfile.NamedTemporaryFile(suffix=suffix) as temporary:
            temporary.write(image)
            temporary.flush()
            return self.backend.recognize(temporary.name, form_text(fields, "language", None))


def main() -> None:
    """Load the OCR models named on the command line and answer on their port."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("paddleocr",), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--host", default=LOOPBACK)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"),
                        default="fp32")
    args = parser.parse_args()
    serve_uploads(OcrHost, PaddleOcrBackend(args.model, args.precision), args.name,
                  (args.host, args.port))


if __name__ == "__main__":
    main()
