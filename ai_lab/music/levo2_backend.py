"""LeVo 2 (SongGeneration 2) through the native LeVo2.cpp runtime.

Each song is one run of `levo-cantor`: lyrics and a style go in, a 48 kHz
stereo WAV comes out. No lyrics means an instrumental. The runtime loads the
model for every song, so this host holds no GPU memory between songs.

Licence: research, academic and education use only. AI-Lab serves it as a
quality reference for tests, never for anything broadcast.
"""
from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from threading import Lock

from .request_fields import (
    MAX_LYRICS_CHARS,
    MAX_PROMPT_CHARS,
    TextRule,
    check_fields,
    flag,
    seconds,
    seed,
    text,
    wav_answer,
)

ALLOWED_FIELDS = {"model", "prompt", "lyrics", "instrumental", "duration", "seed"}
PROMPT = TextRule("prompt must contain 1–4000 characters", MAX_PROMPT_CHARS, shortest=1)
LYRICS = TextRule("lyrics must be text of at most 12000 characters", MAX_LYRICS_CHARS)
DEFAULT_S, SHORTEST_S, LONGEST_S = 60, 5, 300
# The render settings the runtime is run with.
MAX_STEPS = 200
MAX_GUIDANCE = 10.0
# How many lines of a failed run's output are reported.
ERROR_LINES = 5


@dataclass(frozen=True)
class Levo2Files:
    """The runtime and the three model files one song needs.

    `lm`, `flow` and `vae` are file names inside `model_dir`.
    """

    cantor: Path
    model_dir: Path
    lm: str
    flow: str
    vae: str


class Levo2Backend:
    """LeVo 2, run once per song by the native `levo-cantor` runtime."""

    def __init__(self, files: Levo2Files, output_root: Path, steps: int, cfg: float,
                 timeout_s: int = 3600) -> None:
        self.files = {"lm": files.model_dir / files.lm, "dit": files.model_dir / files.flow,
                      "vae": files.model_dir / files.vae}
        if not files.cantor.is_file():
            raise ValueError("levo-cantor is not built")
        missing = [str(path) for path in self.files.values() if not path.is_file()]
        if missing:
            raise ValueError("LeVo 2 files are absent: " + ", ".join(missing))
        if not 1 <= steps <= MAX_STEPS or not 0 < cfg <= MAX_GUIDANCE:
            raise ValueError("LeVo 2 render settings are invalid")
        output_root.mkdir(parents=True, exist_ok=True)
        self.cantor = files.cantor
        self.model_name = files.model_dir.name
        self.output_root = output_root
        self.steps = steps
        self.cfg = cfg
        self.timeout_s = timeout_s
        self.lock = Lock()

    def generate(self, body: dict) -> dict:
        """One song for the request, as base64 16-bit WAV."""
        request = validate(body, self.model_name)
        with tempfile.TemporaryDirectory(dir=self.output_root) as directory:
            root = Path(directory)
            job = {"caption": request["prompt"], "duration": request["duration"],
                   "seed": request["seed"],
                   "flow": {"seed": request["seed"], "euler_steps": self.steps,
                            "cfg_scale": self.cfg}}
            if not request["instrumental"]:
                job["lyrics"] = request["lyrics"]
            (root / "request.json").write_text(json.dumps(job))
            output = root / "song.wav"
            with self.lock:
                run = subprocess.run([
                    str(self.cantor), "--input", str(root / "request.json"),
                    "--checkpoint", str(root / "song.resume"),
                    "--output", str(output),
                    "--lm", str(self.files["lm"]), "--dit", str(self.files["dit"]),
                    "--vae", str(self.files["vae"])],
                    capture_output=True, text=True, timeout=self.timeout_s)
            if run.returncode != 0 or not output.is_file():
                tail = (run.stderr or run.stdout).strip().splitlines()[-ERROR_LINES:]
                raise RuntimeError("LeVo 2 failed: " + " | ".join(tail))
            wav, length = to_pcm16(output)
        return wav_answer(wav, model=self.model_name, seed=request["seed"],
                          duration=length, instrumental=request["instrumental"])


def validate(body: dict, model_name: str) -> dict:
    """The request's fields, checked; ValueError says what is wrong."""
    check_fields(body, ALLOWED_FIELDS, "LeVo 2", model_name)
    prompt = text(body, "prompt", PROMPT)
    instrumental = flag(body, "instrumental", False)
    lyrics = text(body, "lyrics", LYRICS)
    if not instrumental and not lyrics:
        raise ValueError("a song with vocals needs lyrics; send instrumental: true for none")
    return {"prompt": prompt, "lyrics": lyrics, "instrumental": instrumental,
            "duration": seconds(body, DEFAULT_S, SHORTEST_S, LONGEST_S), "seed": seed(body)}


def to_pcm16(path: Path) -> tuple[bytes, float]:
    """LeVo writes 32-bit float WAV; AI-Lab's music contract is 16-bit PCM."""
    import soundfile as sf
    samples, rate = sf.read(path, dtype="float32")
    out = BytesIO()
    sf.write(out, samples.clip(-1, 1), rate, format="WAV", subtype="PCM_16")
    return out.getvalue(), len(samples) / rate
