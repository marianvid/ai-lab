"""LeVo 2 (SongGeneration 2) through the native LeVo2.cpp runtime.

Each song is one run of `levo-cantor`: lyrics and a style go in, a 48 kHz
stereo WAV comes out. No lyrics means an instrumental. The runtime loads the
model for every song, so this host holds no GPU memory between songs.

Licence: research, academic and education use only. AI-Lab serves it as a
quality reference for tests, never for anything broadcast.
"""
from __future__ import annotations

import base64
import json
import secrets
import subprocess
import tempfile
from io import BytesIO
from pathlib import Path
from threading import Lock

ALLOWED_FIELDS = {"model", "prompt", "lyrics", "instrumental", "duration", "seed"}


class Levo2Backend:
    def __init__(self, cantor: Path, model_dir: Path, lm: str, flow: str,
                 vae: str, output_root: Path, steps: int, cfg: float,
                 timeout_s: int = 3600) -> None:
        self.files = {"lm": model_dir / lm, "dit": model_dir / flow,
                      "vae": model_dir / vae}
        if not cantor.is_file():
            raise ValueError("levo-cantor is not built")
        missing = [str(path) for path in self.files.values() if not path.is_file()]
        if missing:
            raise ValueError("LeVo 2 files are absent: " + ", ".join(missing))
        if not 1 <= steps <= 200 or not 0 < cfg <= 10:
            raise ValueError("LeVo 2 render settings are invalid")
        output_root.mkdir(parents=True, exist_ok=True)
        self.cantor = cantor
        self.model_name = model_dir.name
        self.output_root = output_root
        self.steps = steps
        self.cfg = cfg
        self.timeout_s = timeout_s
        self.lock = Lock()

    def generate(self, body: dict) -> dict:
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
                tail = (run.stderr or run.stdout).strip().splitlines()[-5:]
                raise RuntimeError("LeVo 2 failed: " + " | ".join(tail))
            wav, seconds = to_pcm16(output)
        return {"model": self.model_name, "seed": request["seed"],
                "duration": seconds, "instrumental": request["instrumental"],
                "data": [{"mime_type": "audio/wav",
                          "b64_wav": base64.b64encode(wav).decode()}]}


def validate(body: dict, model_name: str) -> dict:
    unknown = set(body) - ALLOWED_FIELDS
    if unknown:
        raise ValueError("Unknown LeVo 2 fields: " + ", ".join(sorted(unknown)))
    if body.get("model") not in (None, model_name):
        raise ValueError(f"this engine serves {model_name}")
    prompt = body.get("prompt", "")
    if not isinstance(prompt, str) or not 1 <= len(prompt.strip()) <= 4000:
        raise ValueError("prompt must contain 1–4000 characters")
    instrumental = body.get("instrumental", False)
    if not isinstance(instrumental, bool):
        raise ValueError("instrumental must be true or false")
    lyrics = body.get("lyrics", "")
    if not isinstance(lyrics, str) or len(lyrics) > 12000:
        raise ValueError("lyrics must be text of at most 12000 characters")
    if not instrumental and not lyrics.strip():
        raise ValueError("a song with vocals needs lyrics; send instrumental: true for none")
    duration = body.get("duration", 60)
    if type(duration) not in (int, float) or not 5 <= duration <= 300:
        raise ValueError("duration must be between 5 and 300 seconds")
    seed = body.get("seed", secrets.randbelow(2**32))
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed must be a 32-bit unsigned integer")
    return {"prompt": prompt.strip(), "lyrics": lyrics.strip(),
            "instrumental": instrumental, "duration": duration, "seed": seed}


def to_pcm16(path: Path) -> tuple[bytes, float]:
    """LeVo writes 32-bit float WAV; AI-Lab's music contract is 16-bit PCM."""
    import soundfile as sf
    samples, rate = sf.read(path, dtype="float32")
    out = BytesIO()
    sf.write(out, samples.clip(-1, 1), rate, format="WAV", subtype="PCM_16")
    return out.getvalue(), len(samples) / rate
