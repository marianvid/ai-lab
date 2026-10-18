#!/usr/bin/env python3
"""OpenAI-shaped HTTP adapter for local speech runtimes.

This file is deliberately executable by a runtime-specific Python environment.
NeMo and ONNX stay isolated from the manager and from vLLM; only this small
HTTP contract is shared between them. The HTTP side is `engine_host.http_host`'s
`UploadHost`; this file says how each kind of backend answers an upload.
"""

from __future__ import annotations

import argparse
import inspect
import sys
import tempfile
from pathlib import Path
from threading import Lock

# This file is launched directly by runtime-specific Python environments.
# Give those environments access to the small adapter beside this script.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ai_lab.audio.aligner import QwenAlignBackend
from ai_lab.engine_host.http_host import Form, UploadHost, form_text, serve_uploads, uploaded_file
from ai_lab.network import LOOPBACK

# What every backend here is fed: mono audio at 16 kHz.
TARGET_RATE = 16000
# The highest sampling rate accepted from an upload.
MAX_RATE = 192000
# A diarization segment is (start, end, speaker).
SEGMENT_PARTS = 3
STEREO_DIMENSIONS = 2


def _torch_device(torch):
    """Use the accelerator present on this host, with CPU as a safe floor."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(getattr(torch, "backends", None), "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _checkpoint(directory: str) -> str:
    path = Path(directory)
    if path.is_file():
        return str(path)
    found = sorted(path.glob("*.nemo"))
    if len(found) != 1:
        raise RuntimeError(f"expected one .nemo checkpoint in {path}, found {len(found)}")
    return str(found[0])


def _vad_samples(samples, sampling_rate: int):
    """Convert an uploaded recording to the mono 16 kHz input Silero expects."""
    import numpy as np

    if not 0 < sampling_rate <= MAX_RATE:
        raise ValueError("Invalid audio sampling rate")
    if getattr(samples, "ndim", 1) == STEREO_DIMENSIONS:
        samples = samples.mean(axis=1)
    if not len(samples):
        raise ValueError("Audio is empty")
    if sampling_rate == TARGET_RATE:
        return samples
    count = max(1, round(len(samples) * TARGET_RATE / sampling_rate))
    positions = np.arange(count, dtype=np.float64) * sampling_rate / TARGET_RATE
    return np.interp(positions, np.arange(len(samples)), samples).astype("float32")


def _nemo_model(model_path: str, precision: str):
    """A NeMo checkpoint restored on the card at `precision`, and torch."""
    import nemo.collections.asr as nemo_asr
    import torch

    checkpoint = _checkpoint(model_path)
    model = nemo_asr.models.ASRModel.restore_from(checkpoint)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
             "fp32": torch.float32}[precision]
    return model.to(device="cuda", dtype=dtype).eval(), torch, checkpoint


class NemoBackend:
    """Transcription with a NeMo model (Canary, Nemotron, Parakeet)."""

    owner = "nemo"
    path = "/v1/audio/transcriptions"

    def __init__(self, model_path: str, precision: str) -> None:
        self.model, self.torch, checkpoint = _nemo_model(model_path, precision)
        name = f"{model_path} {checkpoint} {self.model.__class__.__name__}".lower()
        self.model_kind = next((kind for kind in ("canary", "nemotron", "parakeet")
                                if kind in name), "generic")
        self.lock = Lock()

    def transcribe(self, audio_path: str, language: str | None = None) -> str:
        """The text of one recording; `language` as the model family spells it."""
        with self.lock, self.torch.inference_mode():
            result = self.model.transcribe([audio_path], **self._options(language))[0]
        text = getattr(result, "text", result)
        return str(text or "")

    def _options(self, language: str | None) -> dict:
        """What this model family's `transcribe` takes for the language, if anything."""
        parameters = inspect.signature(self.model.transcribe).parameters
        open_ended = any(parameter.kind is inspect.Parameter.VAR_KEYWORD
                         for parameter in parameters.values())
        wanted = _language_options(self.model_kind, language)
        return {key: value for key, value in wanted.items()
                if key in parameters or (open_ended and key != "language")}


def _language_options(model_kind: str, language: str | None) -> dict:
    """The language options each NeMo family understands, before checking the signature.

    Canary always gets its source and target language (Romanian unless told)
    and its task; Nemotron a BCP 47 target language; the others a plain
    `language`, only when the model's `transcribe` names that parameter.
    """
    if model_kind == "canary":
        return {"source_lang": language or "ro", "target_lang": language or "ro",
                "task": "asr", "pnc": "yes"}
    if not language:
        return {}
    if model_kind == "nemotron":
        return {"target_lang": "ro-RO" if language == "ro" else language}
    return {"language": language}


class MlxWhisperBackend:
    """Transcription with Whisper on Apple silicon (mlx-whisper)."""

    owner = "mlx-whisper"
    path = "/v1/audio/transcriptions"

    def __init__(self, model_path: str, _precision: str = "") -> None:
        import mlx_whisper
        self.mlx_whisper = mlx_whisper
        self.model_path = model_path
        self.lock = Lock()

    def transcribe(self, audio_path: str, language: str | None = None) -> str:
        """The text of one recording."""
        options = {"path_or_hf_repo": self.model_path}
        if language:
            options["language"] = language
        with self.lock:
            return self.mlx_whisper.transcribe(audio_path, **options)["text"]


class SortformerBackend:
    """Who speaks when, with NeMo's Sortformer."""

    owner = "nemo-sortformer"
    path = "/v1/audio/diarizations"

    def __init__(self, model_path: str, precision: str) -> None:
        self.model, self.torch, _ = _nemo_model(model_path, precision)
        self.lock = Lock()

    @staticmethod
    def _segment(value) -> dict:
        parts = value.split() if isinstance(value, str) else list(value)
        if len(parts) != SEGMENT_PARTS:
            raise ValueError(f"unexpected diarization segment: {value!r}")
        start, end, speaker = parts
        speaker = str(speaker)
        if speaker.isdigit():
            speaker = f"speaker_{speaker}"
        return {"start": float(start), "end": float(end), "speaker": speaker}

    def diarize(self, audio_path: str) -> list[dict]:
        """The recording's speaker turns: start, end (seconds) and speaker."""
        with self.lock, self.torch.inference_mode():
            result = self.model.diarize([audio_path], batch_size=1,
                                        verbose=False)[0]
        return [self._segment(segment) for segment in result]


class PyannoteBackend:
    """Who speaks when, with a pyannote.audio pipeline."""

    owner = "pyannote"
    path = "/v1/audio/diarizations"

    def __init__(self, model_path: str, precision: str) -> None:
        import torch
        from pyannote.audio import Pipeline

        self.torch = torch
        self.pipeline = Pipeline.from_pretrained(model_path)
        self.pipeline.to(_torch_device(torch))
        self.lock = Lock()

    def diarize(self, audio_path: str) -> list[dict]:
        """The recording's speaker turns: start, end (seconds) and speaker."""
        with self.lock, self.torch.inference_mode():
            output = self.pipeline(audio_path)
        annotation = getattr(output, "speaker_diarization", output)
        return [{"start": float(turn.start), "end": float(turn.end),
                 "speaker": str(speaker)}
                for turn, _, speaker in annotation.itertracks(yield_label=True)]


class SileroBackend:
    """Where speech is, with Silero VAD (ONNX)."""

    owner = "silero-vad"
    path = "/v1/audio/speech-segments"

    def __init__(self, model_path: str, precision: str = "bf16") -> None:
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
        """The stretches of speech in the recording, in seconds."""
        samples, sampling_rate = self.soundfile.read(audio_path, dtype="float32")
        audio = self.torch.from_numpy(_vad_samples(samples, sampling_rate))
        with self.lock:
            return self.get_speech_timestamps(
                audio, self.model, sampling_rate=TARGET_RATE,
                threshold=threshold, min_silence_duration_ms=min_silence_ms,
                return_seconds=True)


AudioBackend = (NemoBackend | MlxWhisperBackend | SortformerBackend | PyannoteBackend
                | SileroBackend | QwenAlignBackend)
BACKENDS: dict[str, type[AudioBackend]] = {
    "nemo": NemoBackend, "mlx-whisper": MlxWhisperBackend,
    "sortformer": SortformerBackend, "pyannote": PyannoteBackend,
    "silero": SileroBackend, "qwen-align": QwenAlignBackend}


class AudioHost(UploadHost):
    """Answers an uploaded recording with what this process's backend does."""

    backend: AudioBackend

    def answer(self, fields: Form) -> dict:
        """The recording, saved to a temporary file and handed to the backend."""
        filename, audio = uploaded_file(fields, "audio")
        suffix = Path(filename or "audio.wav").suffix or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix) as temporary:
            temporary.write(audio)
            temporary.flush()
            return _answer(self.backend, temporary.name, fields)


def _answer(backend: AudioBackend, audio_path: str, fields: Form) -> dict:
    """What each kind of backend makes of one recording."""
    if isinstance(backend, (NemoBackend, MlxWhisperBackend)):
        return {"text": backend.transcribe(audio_path, form_text(fields, "language", None))}
    if isinstance(backend, (SortformerBackend, PyannoteBackend)):
        segments = backend.diarize(audio_path)
        return {"segments": segments,
                "speakers": sorted({item["speaker"] for item in segments})}
    if isinstance(backend, QwenAlignBackend):
        return {"words": backend.align(audio_path, form_text(fields, "text", "") or "",
                                       form_text(fields, "language", "English") or "English")}
    threshold = float(form_text(fields, "threshold", None) or 0.5)
    silence = int(form_text(fields, "min_silence_duration_ms", None) or 100)
    return {"segments": backend.segments(audio_path, threshold, silence)}


def main() -> None:
    """Load the backend named on the command line and answer on its port."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=tuple(BACKENDS), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--host", default=LOOPBACK)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"),
                        default="bf16")
    args = parser.parse_args()
    backend = BACKENDS[args.backend](args.model, args.precision)
    serve_uploads(AudioHost, backend, args.name, (args.host, args.port))


if __name__ == "__main__":
    main()
