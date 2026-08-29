#!/usr/bin/env python3
"""HTTP adapter for a local PaddleOCR/PaddleX pipeline.

Executable by a runtime-specific Python environment, the same way
`ai_lab/audio/server.py` is: PaddleOCR stays isolated from the manager, and
only this small HTTP contract is shared with it.

The response is AI-Lab's own stable OCR schema, not PaddleOCR's native
output shape — Data-Lab and every other client depend on this shape, not on
whatever PaddleOCR happens to return this version.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from email import policy
from email.parser import BytesParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock


class PaddleOcrBackend:
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
        with self.lock:
            result = list(self.pipeline.predict(image_path))
        lines = []
        orientation = 0
        for page in result:
            payload = getattr(page, "json", page)
            if callable(payload):
                payload = payload()
            if isinstance(payload, str):
                payload = json.loads(payload)
            if isinstance(payload, dict) and isinstance(payload.get("res"), dict):
                payload = payload["res"]
            if not isinstance(payload, dict):
                continue
            texts = payload.get("rec_texts", [])
            scores = payload.get("rec_scores", [])
            polygons = payload.get("rec_polys") or payload.get("dt_polys") or []
            for index, text in enumerate(texts):
                polygon = polygons[index] if index < len(polygons) else []
                score = scores[index] if index < len(scores) else 0.0
                lines.append({"text": str(text), "confidence": float(score),
                              "polygon": [[float(x), float(y)] for x, y in polygon]})
            orientation = int(payload.get("doc_preprocessor_res", {}).get(
                "angle", orientation) or orientation)
        text = "\n".join(line["text"] for line in lines)
        return {
            "text": text,
            "lines": lines,
            "language": language or "",
            "orientation_degrees": orientation,
            "warnings": [] if lines else ["no text recognized"],
        }

class Handler(BaseHTTPRequestHandler):
    backend = None
    model_name = ""

    def do_GET(self):
        if self.path == "/health":
            self._json(200, {"status": "ok"})
        elif self.path == "/v1/models":
            self._json(200, {"object": "list", "data": [{
                "id": self.model_name, "object": "model",
                "owned_by": self.backend.owner
            }]})
        else:
            self._json(404, {"error": {"message": "not found"}})

    def do_POST(self):
        if self.path != self.backend.path:
            self._json(404, {"error": {"message": "not found"}})
            return
        try:
            fields = self._multipart()
            uploaded = fields.get("file")
            if uploaded is None:
                raise ValueError("the request must contain an image file")
            filename, image = uploaded
            suffix = Path(filename or "image.png").suffix or ".png"
            language_part = fields.get("language")
            language = (language_part[1].decode("utf-8")
                        if language_part is not None else None)
            with tempfile.NamedTemporaryFile(suffix=suffix) as temporary:
                temporary.write(image)
                temporary.flush()
                result = self.backend.recognize(temporary.name, language)
            self._json(200, result)
        except Exception as error:
            self._json(400, {"error": {"message": str(error),
                                       "type": error.__class__.__name__}})

    def log_message(self, format, *args):
        print(f"{self.address_string()} {format % args}", flush=True)

    def _multipart(self) -> dict[str, tuple[str | None, bytes]]:
        length = int(self.headers.get("Content-Length", "0"))
        content_type = self.headers.get("Content-Type", "")
        if not content_type.lower().startswith("multipart/form-data"):
            raise ValueError("expected a multipart/form-data request")
        envelope = (f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode()
                    + self.rfile.read(length))
        message = BytesParser(policy=policy.default).parsebytes(envelope)
        fields = {}
        for part in message.iter_parts():
            name = part.get_param("name", header="content-disposition")
            if name:
                fields[name] = (part.get_filename(), part.get_payload(decode=True) or b"")
        return fields

    def _json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("paddleocr",), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"),
                        default="fp32")
    args = parser.parse_args()
    Handler.backend = PaddleOcrBackend(args.model, args.precision)
    Handler.model_name = args.name
    print(f"{args.backend} ready with {args.name} on {args.host}:{args.port}", flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
