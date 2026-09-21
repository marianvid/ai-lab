"""AI-Lab music contract backed by the resident ds-yue-webui worker."""
from __future__ import annotations

import base64
import json
import secrets
import time
import urllib.error
import urllib.request
from io import BytesIO
from pathlib import Path


class Yue2WebBackend:
    def __init__(self, ui_url: str, model_name: str, cot: str) -> None:
        self.ui_url = ui_url.rstrip("/")
        self.model_name = model_name
        self.cot = cot

    def generate(self, body: dict) -> dict:
        request = self._validate(body)
        params = {"style": request["prompt"], "lyrics": request["lyrics"],
                  "cot": self.cot, "seed": request["seed"]}
        if request["abc"]:
            params["abc"] = request["abc"]
        response = self._json("POST", "/api/jobs", {
            "kind": "generate", "name": "AI-Lab generation", "params": params})
        job_id = response["job"]["id"]
        deadline = time.monotonic() + 3600
        while time.monotonic() < deadline:
            job = self._json("GET", f"/api/jobs/{job_id}")
            if job["status"] == "done":
                break
            if job["status"] in {"failed", "cancelled"}:
                raise RuntimeError(job.get("error") or f"YuE2 job {job['status']}")
            time.sleep(1)
        else:
            raise TimeoutError("YuE2 generation exceeded one hour")
        output = Path(job["output_dir"])
        audio_path = output / "audio.flac"
        if not audio_path.is_file():
            raise RuntimeError("YuE2 completed without audio.flac")
        import soundfile as sf
        samples, sample_rate = sf.read(audio_path, dtype="float32")
        wav = BytesIO()
        sf.write(wav, samples, sample_rate, format="WAV", subtype="PCM_16")
        score = output / "score.abc"
        result = job.get("result") or {}
        return {"model": self.model_name, "seed": request["seed"],
                "duration": result.get("audio_seconds", len(samples) / sample_rate),
                "truncated": bool(result.get("truncated")),
                "score_abc": score.read_text() if score.is_file() else "",
                "data": [{"mime_type": "audio/wav",
                          "b64_wav": base64.b64encode(wav.getvalue()).decode()}]}

    def _json(self, method: str, path: str, body: dict | None = None) -> dict:
        payload = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(
            self.ui_url + path, data=payload, method=method,
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            message = error.read().decode(errors="replace")
            raise RuntimeError(f"YuE2 Studio rejected the request: {message}") from error

    def _validate(self, body: dict) -> dict:
        allowed = {"model", "prompt", "lyrics", "instrumental", "seed", "abc"}
        unknown = set(body) - allowed
        if unknown:
            raise ValueError("Unknown YuE2 fields: " + ", ".join(sorted(unknown)))
        if body.get("model") not in (None, self.model_name):
            raise ValueError(f"this engine serves {self.model_name}")
        prompt, lyrics, abc = (body.get("prompt", ""), body.get("lyrics", ""),
                               body.get("abc", ""))
        if not isinstance(prompt, str) or not 1 <= len(prompt.strip()) <= 4000:
            raise ValueError("style must contain 1–4000 characters")
        if not isinstance(lyrics, str) or not 1 <= len(lyrics.strip()) <= 12000:
            raise ValueError("lyrics must contain 1–12000 characters")
        if not isinstance(abc, str) or len(abc) > 40000:
            raise ValueError("ABC score is too large")
        if body.get("instrumental") is not False:
            raise ValueError("YuE2 needs lyric-conditioned generation")
        seed = body.get("seed", secrets.randbelow(2**32))
        if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
            raise ValueError("seed must be a 32-bit unsigned integer")
        return {"prompt": prompt.strip(), "lyrics": lyrics.strip(),
                "abc": abc, "seed": seed}
