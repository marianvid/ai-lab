"""HeartMuLa pipeline with a bounded request and WAV result contract."""
from __future__ import annotations

import tempfile
from pathlib import Path
from threading import Lock

from ..bundles import Checkpoint, Sampling
from ..request_fields import (
    MAX_LYRICS_CHARS,
    MAX_PROMPT_CHARS,
    TextRule,
    check_fields,
    checked_wav,
    seconds,
    seed,
    text,
    wav_answer,
    wav_length_s,
)

ALLOWED_FIELDS = {"model", "prompt", "lyrics", "instrumental", "duration", "seed"}
_NEEDS_BOTH = "HeartMuLa needs style tags and lyrics"
PROMPT = TextRule(_NEEDS_BOTH, MAX_PROMPT_CHARS, shortest=1, coerce=True)
LYRICS = TextRule(_NEEDS_BOTH, MAX_LYRICS_CHARS, shortest=1, coerce=True)
DEFAULT_S, SHORTEST_S, LONGEST_S = 60, 5, 180
MS_PER_S = 1000


class HeartMulaBackend:
    """HeartMuLa 3B: a song from style tags and lyrics, the model kept loaded."""

    def __init__(self, checkpoint: Checkpoint, output_root: Path, version: str,
                 sampling: Sampling) -> None:
        import torch
        from heartlib import HeartMuLaGenPipeline

        checkpoint.check("HeartMuLa")
        if not version:
            raise ValueError("HeartMuLa inference settings are invalid")
        sampling.check("HeartMuLa")
        output_root.mkdir(parents=True, exist_ok=True)
        self.output_root = output_root
        self.model_name = checkpoint.checkpoint_path.name
        self.sampling = sampling
        self.torch = torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pipeline = HeartMuLaGenPipeline.from_pretrained(
            str(checkpoint.bundle_root), device={"mula": device, "codec": device},
            dtype={"mula": torch.bfloat16 if device.type == "cuda"
                   else torch.float32, "codec": torch.float32},
            version=version, lazy_load=True)
        self.lock = Lock()

    def generate(self, body: dict) -> dict:
        """One song for the request, as base64 WAV."""
        request = self._validate(body)
        with tempfile.TemporaryDirectory(dir=self.output_root) as directory:
            root = Path(directory)
            lyrics_path, tags_path = root / "lyrics.txt", root / "tags.txt"
            output = root / "result.wav"
            lyrics_path.write_text(request["lyrics"] + "\n")
            tags_path.write_text(request["prompt"] + "\n")
            with self.lock, self.torch.no_grad():
                self.torch.manual_seed(request["seed"])
                if self.torch.cuda.is_available():
                    self.torch.cuda.manual_seed_all(request["seed"])
                self.pipeline(
                    {"lyrics": str(lyrics_path), "tags": str(tags_path)},
                    max_audio_length_ms=round(request["duration"] * MS_PER_S),
                    save_path=str(output), topk=self.sampling.topk,
                    temperature=self.sampling.temperature,
                    cfg_scale=self.sampling.cfg_scale)
            audio = checked_wav(output.read_bytes(), "HeartMuLa")
            length = wav_length_s(output)
        return wav_answer(audio, model=self.model_name, seed=request["seed"],
                          duration=length)

    def _validate(self, body: dict) -> dict:
        check_fields(body, ALLOWED_FIELDS, "HeartMuLa", self.model_name)
        prompt = text(body, "prompt", PROMPT)
        lyrics = text(body, "lyrics", LYRICS)
        if body.get("instrumental") is not False:
            raise ValueError("HeartMuLa needs lyric-conditioned generation")
        return {"prompt": prompt, "lyrics": lyrics,
                "duration": seconds(body, DEFAULT_S, SHORTEST_S, LONGEST_S),
                "seed": seed(body)}
