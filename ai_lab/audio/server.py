#!/usr/bin/env python3
"""OpenAI-shaped HTTP adapter for local speech runtimes.

This file is deliberately executable by a runtime-specific Python environment.
NeMo and ONNX stay isolated from the manager and from vLLM; only this small
HTTP contract is shared between them.
"""

from __future__ import annotations

import argparse
import inspect
import json
import tempfile
from email import policy
from email.parser import BytesParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock


def _checkpoint(directory: str) -> str:
    path = Path(directory)
    if path.is_file():
        return str(path)
    found = sorted(path.glob("*.nemo"))
    if len(found) != 1:
        raise RuntimeError(f"expected one .nemo checkpoint in {path}, found {len(found)}")
    return str(found[0])


class NemoBackend:
    owner = "nemo"
    path = "/v1/audio/transcriptions"
    def __init__(self, model_path: str, precision: str) -> None:
        import torch
        import nemo.collections.asr as nemo_asr

        self.torch = torch
        checkpoint = _checkpoint(model_path)
        self.model = nemo_asr.models.ASRModel.restore_from(checkpoint)
        name = f"{model_path} {checkpoint} {self.model.__class__.__name__}".lower()
        self.model_kind = next((kind for kind in ("canary", "nemotron", "parakeet")
                                if kind in name), "generic")
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
                 "fp32": torch.float32}[precision]
        self.model = self.model.to(device="cuda", dtype=dtype).eval()
        self.lock = Lock()

    def transcribe(self, audio_path: str, language: str | None = None) -> str:
        kwargs = {}
        parameters = inspect.signature(self.model.transcribe).parameters
        accepts_prompt = any(parameter.kind is inspect.Parameter.VAR_KEYWORD
                             for parameter in parameters.values())
        if self.model_kind == "canary":
            for key, value in (("source_lang", language or "ro"),
                               ("target_lang", language or "ro"),
                               ("task", "asr"), ("pnc", "yes")):
                if key in parameters or accepts_prompt:
                    kwargs[key] = value
        elif (self.model_kind == "nemotron"
              and ("target_lang" in parameters or accepts_prompt) and language):
            kwargs["target_lang"] = "ro-RO" if language == "ro" else language
        elif "language" in parameters and language:
            kwargs["language"] = language
        with self.lock, self.torch.inference_mode():
            result = self.model.transcribe([audio_path], **kwargs)[0]
        text = getattr(result, "text", result)
        return str(text or "")


class SortformerBackend:
    owner = "nemo-sortformer"
    path = "/v1/audio/diarizations"

    def __init__(self, model_path: str, precision: str) -> None:
        import torch
        import nemo.collections.asr as nemo_asr

        self.torch = torch
        self.model = nemo_asr.models.ASRModel.restore_from(_checkpoint(model_path))
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
                 "fp32": torch.float32}[precision]
        self.model = self.model.to(device="cuda", dtype=dtype).eval()
        self.lock = Lock()

    @staticmethod
    def _segment(value) -> dict:
        if isinstance(value, str):
            parts = value.split()
        else:
            parts = list(value)
        if len(parts) != 3:
            raise ValueError(f"unexpected diarization segment: {value!r}")
        start, end, speaker = parts
        speaker = str(speaker)
        if speaker.isdigit():
            speaker = f"speaker_{speaker}"
        return {"start": float(start), "end": float(end), "speaker": speaker}

    def diarize(self, audio_path: str) -> list[dict]:
        with self.lock, self.torch.inference_mode():
            result = self.model.diarize([audio_path], batch_size=1,
                                        verbose=False)[0]
        return [self._segment(segment) for segment in result]


class PyannoteBackend:
    owner = "pyannote"
    path = "/v1/audio/diarizations"

    def __init__(self, model_path: str, precision: str) -> None:
        import torch
        from pyannote.audio import Pipeline

        self.torch = torch
        self.pipeline = Pipeline.from_pretrained(model_path)
        self.pipeline.to(torch.device("cuda"))
        self.lock = Lock()

    def diarize(self, audio_path: str) -> list[dict]:
        with self.lock, self.torch.inference_mode():
            output = self.pipeline(audio_path)
        annotation = getattr(output, "speaker_diarization", output)
        return [{"start": float(turn.start), "end": float(turn.end),
                 "speaker": str(speaker)}
                for turn, _, speaker in annotation.itertracks(yield_label=True)]


class SileroBackend:
    owner = "silero-vad"
    path = "/v1/audio/speech-segments"

    def __init__(self, model_path: str, precision: str = "bf16") -> None:
        import sys
        checkpoint = Path(model_path).resolve()
        repository = next(
            (candidate for candidate in (checkpoint, *checkpoint.parents)
             if (candidate / "src" / "silero_vad").is_dir()), None)
        if repository is None:
            raise RuntimeError(f"Silero source tree not found above {checkpoint}")
        sys.path.insert(0, str(repository / "src"))
        import soundfile
        import torch
        from silero_vad import get_speech_timestamps, load_silero_vad
        self.get_speech_timestamps = get_speech_timestamps
        self.soundfile = soundfile
        self.torch = torch
        self.model = load_silero_vad(onnx=True)
        self.lock = Lock()

    def segments(self, audio_path: str, threshold: float = 0.5,
                 min_silence_ms: int = 100) -> list[dict]:
        samples, sampling_rate = self.soundfile.read(audio_path, dtype="float32")
        if sampling_rate != 16000:
            raise ValueError("Silero VAD expects audio normalized to 16 kHz")
        if getattr(samples, "ndim", 1) == 2:
            samples = samples.mean(axis=1)
        audio = self.torch.from_numpy(samples)
        with self.lock:
            return self.get_speech_timestamps(
                audio, self.model, sampling_rate=16000,
                threshold=threshold, min_silence_duration_ms=min_silence_ms,
                return_seconds=True)


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
                raise ValueError("the request must contain an audio file")
            filename, audio = uploaded
            suffix = Path(filename or "audio.wav").suffix or ".wav"
            with tempfile.NamedTemporaryFile(suffix=suffix) as temporary:
                temporary.write(audio)
                temporary.flush()
                if isinstance(self.backend, NemoBackend):
                    language_part = fields.get("language")
                    language = (language_part[1].decode("utf-8")
                                if language_part is not None else None)
                    result = {"text": self.backend.transcribe(temporary.name, language)}
                elif isinstance(self.backend, (SortformerBackend, PyannoteBackend)):
                    segments = self.backend.diarize(temporary.name)
                    result = {"segments": segments,
                              "speakers": sorted({item["speaker"]
                                                  for item in segments})}
                else:
                    threshold = self._number(fields, "threshold", 0.5, float)
                    silence = self._number(fields, "min_silence_duration_ms", 100, int)
                    result = {"segments": self.backend.segments(
                        temporary.name, threshold, silence)}
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

    @staticmethod
    def _number(fields, name, default, converter):
        part = fields.get(name)
        return converter(part[1].decode("utf-8")) if part is not None else default

    def _json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend",
                        choices=("nemo", "sortformer", "pyannote", "silero"),
                        required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"),
                        default="bf16")
    args = parser.parse_args()
    backends = {"nemo": NemoBackend, "sortformer": SortformerBackend,
                "pyannote": PyannoteBackend, "silero": SileroBackend}
    backend = backends[args.backend]
    Handler.backend = backend(args.model, args.precision)
    Handler.model_name = args.name
    print(f"{args.backend} ready with {args.name} on {args.host}:{args.port}", flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
