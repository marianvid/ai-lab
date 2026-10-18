"""Supervised Khala CLI invocation with the AI-Lab music result contract."""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from threading import Lock

from .request_fields import (
    MAX_LYRICS_CHARS,
    MAX_PROMPT_CHARS,
    TextRule,
    check_fields,
    checked_wav,
    flag,
    seed,
    text,
    wav_answer,
    wav_length_s,
)

ALLOWED_FIELDS = {"model", "prompt", "lyrics", "instrumental", "seed", "length_bucket",
                  "vocal_language"}
_BAD_LENGTH = "Khala prompt or lyrics length is invalid"
PROMPT = TextRule(_BAD_LENGTH, MAX_PROMPT_CHARS, shortest=1, coerce=True)
LYRICS = TextRule(_BAD_LENGTH, MAX_LYRICS_CHARS, coerce=True)
LANGUAGES = {"Chinese", "English", "Japanese", "Korean", "Cantonese"}
# Khala's lengths are numbered buckets, 0 (shortest) to this.
LONGEST_BUCKET = 20
# How long one song may take on the Mac before it is given up.
SONG_LIMIT_S = 1200
# How much of a failed run's error output is reported.
ERROR_TAIL_CHARS = 700


class KhalaBackend:
    """Khala, run as its own command line for every song (the model is not kept loaded)."""

    def __init__(self, generator_script: Path, model_path: Path,
                 output_root: Path, buckets: tuple[int, int]) -> None:
        """`buckets` is (the default length bucket, the longest one allowed)."""
        default_bucket, maximum_bucket = buckets
        if not generator_script.is_file() or not model_path.is_dir():
            raise ValueError("Khala generator or checkpoint is absent")
        if not 0 <= default_bucket <= maximum_bucket <= LONGEST_BUCKET:
            raise ValueError("Khala length buckets are invalid")
        output_root.mkdir(parents=True, exist_ok=True)
        self.script = generator_script
        self.model_path = model_path
        self.output_root = output_root
        self.model_name = model_path.name
        self.default_bucket = default_bucket
        self.maximum_bucket = maximum_bucket
        self.lock = Lock()

    def generate(self, body: dict) -> dict:
        """One song for the request, as base64 WAV."""
        request = self._validate(body)
        with tempfile.TemporaryDirectory(dir=self.output_root) as directory:
            output = Path(directory) / "result.wav"
            environment = os.environ.copy()
            environment.update({"KHALA_BACKEND": "vanilla",
                                "KHALA_VANILLA_WEIGHTS": str(self.model_path)})
            with self.lock:
                completed = subprocess.run(
                    self._command(request, output), cwd=self.script.parent.parent,
                    env=environment, capture_output=True, text=True, timeout=SONG_LIMIT_S)
            if completed.returncode:
                raise RuntimeError("Khala generation failed: " +
                                   completed.stderr[-ERROR_TAIL_CHARS:])
            audio = checked_wav(output.read_bytes(), "Khala")
            length = wav_length_s(output)
        return wav_answer(audio, model=self.model_name, seed=request["seed"],
                          duration=length, length_bucket=request["length_bucket"])

    def _command(self, request: dict, output: Path) -> list[str]:
        """Khala's command line for one song."""
        command = [
            sys.executable, "-u", str(self.script),
            "--language", "Instrumental" if request["instrumental"]
            else request["vocal_language"],
            "--description", request["prompt"],
            "--duration", str(request["length_bucket"]),
            "--seed", str(request["seed"]), "--out", str(output)]
        if not request["instrumental"]:
            command += ["--lyrics", request["lyrics"]]
        return command

    def _validate(self, body: dict) -> dict:
        check_fields(body, ALLOWED_FIELDS, "Khala", self.model_name)
        prompt = text(body, "prompt", PROMPT)
        lyrics = text(body, "lyrics", LYRICS)
        language = str(body.get("vocal_language", "English"))
        if language not in LANGUAGES:
            raise ValueError("Unsupported Khala vocal language")
        return {"prompt": prompt, "lyrics": lyrics,
                "instrumental": flag(body, "instrumental", not bool(lyrics)),
                "vocal_language": language, "length_bucket": self._bucket(body),
                "seed": seed(body)}

    def _bucket(self, body: dict) -> int:
        bucket = body.get("length_bucket", self.default_bucket)
        if type(bucket) is not int or not 0 <= bucket <= self.maximum_bucket:
            raise ValueError("Khala length bucket is out of range")
        return bucket
