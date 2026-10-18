"""MuLaCover generation from a bounded user-supplied WAV reference."""
from __future__ import annotations

import tempfile
from pathlib import Path
from threading import Lock

from ..bundles import Checkpoint, Sampling
from ..request_fields import (
    MAX_LYRICS_CHARS,
    MAX_PROMPT_CHARS,
    TextRule,
    WavRule,
    check_fields,
    checked_wav,
    decoded_wav,
    seed,
    text,
    wav_answer,
    wav_length_s,
)

MAX_SOURCE_BYTES = 25 * 1024 * 1024
ALLOWED_FIELDS = {"model", "prompt", "lyrics", "instrumental", "seed", "reference_audio_base64"}
PROMPT = TextRule("style tags must contain 1–4000 characters", MAX_PROMPT_CHARS, shortest=1)
LYRICS = TextRule("lyrics must contain 1–12000 characters", MAX_LYRICS_CHARS, shortest=1)
REFERENCE = WavRule(MAX_SOURCE_BYTES, "reference WAV is missing or too large",
                    "reference audio is not valid base64",
                    "reference audio must be a WAV of at most 25 MiB")


class MulaCoverBackend:
    """MuLaCover: a cover of a reference recording, in new style tags with given lyrics."""

    def __init__(self, checkpoint: Checkpoint, output_root: Path, sampling: Sampling) -> None:
        import torch
        from mulacover import MuLaCoverGenPipeline

        checkpoint.check("MuLaCover")
        sampling.check("MuLaCover")
        output_root.mkdir(parents=True, exist_ok=True)
        self.output_root = output_root
        self.model_name = checkpoint.checkpoint_path.name
        self.sampling = sampling
        self.torch = torch
        device = torch.device("cuda:0")
        self.pipeline = MuLaCoverGenPipeline.from_pretrained(
            str(checkpoint.bundle_root), device=device,
            dtype={"mulacover": torch.bfloat16, "codec": torch.float32,
                   "qwen": torch.float32, "transcriptor": torch.float32},
            lazy_load=True)
        self.lock = Lock()

    def generate(self, body: dict) -> dict:
        """One cover for the request, as base64 WAV."""
        request = self._validate(body)
        with tempfile.TemporaryDirectory(dir=self.output_root) as directory:
            root = Path(directory)
            reference, lyrics = root / "reference.wav", root / "lyrics.txt"
            tags, output = root / "tags.txt", root / "cover.wav"
            reference.write_bytes(request["audio"])
            lyrics.write_text(request["lyrics"] + "\n")
            tags.write_text(request["prompt"] + "\n")
            with self.lock, self.torch.no_grad():
                self.torch.manual_seed(request["seed"])
                self.torch.cuda.manual_seed_all(request["seed"])
                self.pipeline(
                    {"ref_audio": str(reference), "lyrics": str(lyrics),
                     "tags": str(tags)}, save_path=str(output),
                    topk=self.sampling.topk, temperature=self.sampling.temperature,
                    cfg_scale=self.sampling.cfg_scale)
            audio = checked_wav(output.read_bytes(), "MuLaCover")
            return wav_answer(audio, model=self.model_name, seed=request["seed"],
                              duration=wav_length_s(output))

    def _validate(self, body: dict) -> dict:
        check_fields(body, ALLOWED_FIELDS, "MuLaCover", self.model_name)
        prompt = text(body, "prompt", PROMPT)
        lyrics = text(body, "lyrics", LYRICS)
        if body.get("instrumental") is not False:
            raise ValueError("MuLaCover requires a vocal arrangement")
        audio = decoded_wav(body.get("reference_audio_base64", ""), REFERENCE)
        return {"prompt": prompt, "lyrics": lyrics, "audio": audio, "seed": seed(body)}
