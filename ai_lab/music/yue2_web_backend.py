"""AI-Lab music contract backed by the resident ds-yue-webui worker."""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from io import BytesIO
from pathlib import Path

from .request_fields import (
    MAX_LYRICS_CHARS,
    MAX_PROMPT_CHARS,
    TextRule,
    check_fields,
    seed,
    text,
    wav_answer,
)

ALLOWED_FIELDS = {"model", "prompt", "lyrics", "instrumental", "seed", "abc"}
PROMPT = TextRule("style must contain 1–4000 characters", MAX_PROMPT_CHARS, shortest=1)
LYRICS = TextRule("lyrics must contain 1–12000 characters", MAX_LYRICS_CHARS, shortest=1)
# An editable ABC score sent back with a request; longer is not a score.
MAX_SCORE_CHARS = 40000
# How long one song may take, and how often the studio is asked about it.
SONG_LIMIT_S = 3600
POLL_S = 1.0
# How long one call to the studio may take.
CALL_S = 30


def was_truncated(value) -> bool:
    """YuE2 reports truncation per stage, as {"abc": False, "semantic": False}.

    A non-empty dictionary is truthy in Python, so `bool()` of that report
    said "truncated" for every song, including ones that finished cleanly.
    A song is truncated only when some stage says so.
    """
    if isinstance(value, dict):
        return any(bool(stage) for stage in value.values())
    return bool(value)


class Yue2WebBackend:
    """YuE2, through the ds-yue-webui studio that keeps the model loaded.

    A request becomes one job in the studio; this waits for it and returns the
    song, the studio's ABC score, and whether a stage was cut short.
    """

    def __init__(self, ui_url: str, model_name: str, cot: str) -> None:
        self.ui_url = ui_url.rstrip("/")
        self.model_name = model_name
        self.cot = cot

    def generate(self, body: dict) -> dict:
        """One song for the request, as base64 WAV, with its score."""
        request = self._validate(body)
        params = {"style": request["prompt"], "lyrics": request["lyrics"],
                  "cot": self.cot, "seed": request["seed"]}
        if request["abc"]:
            params["abc"] = request["abc"]
        response = self._json("POST", "/api/jobs", {
            "kind": "generate", "name": "AI-Lab generation", "params": params})
        job_id = response["job"]["id"]
        deadline = time.monotonic() + SONG_LIMIT_S
        while time.monotonic() < deadline:
            job = self._json("GET", f"/api/jobs/{job_id}")
            if job["status"] == "done":
                break
            if job["status"] in {"failed", "cancelled"}:
                raise RuntimeError(job.get("error") or f"YuE2 job {job['status']}")
            time.sleep(POLL_S)
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
        return wav_answer(
            wav.getvalue(), model=self.model_name, seed=request["seed"],
            duration=result.get("audio_seconds", len(samples) / sample_rate),
            truncated=was_truncated(result.get("truncated")),
            score_abc=score.read_text() if score.is_file() else "")

    def _json(self, method: str, path: str, body: dict | None = None) -> dict:
        payload = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(
            self.ui_url + path, data=payload, method=method,
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=CALL_S) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            message = error.read().decode(errors="replace")
            raise RuntimeError(f"YuE2 Studio rejected the request: {message}") from error

    def _validate(self, body: dict) -> dict:
        check_fields(body, ALLOWED_FIELDS, "YuE2", self.model_name)
        prompt = text(body, "prompt", PROMPT)
        lyrics = text(body, "lyrics", LYRICS)
        abc = body.get("abc", "")
        if not isinstance(abc, str) or len(abc) > MAX_SCORE_CHARS:
            raise ValueError("ABC score is too large")
        if body.get("instrumental") is not False:
            raise ValueError("YuE2 needs lyric-conditioned generation")
        return {"prompt": prompt, "lyrics": lyrics, "abc": abc, "seed": seed(body)}
