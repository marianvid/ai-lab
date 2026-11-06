"""AI-Lab music contract backed by the resident ds-yue-webui worker.

Two calls: a song (`generate`), and one window of an earlier song regenerated
(`generate_window`, `docs/music.md`). The window's work is YuE2 Studio's `window`
job (the patch in `system/engines/`); what is here is what AI-Lab owns: the
request's shape, its refusals (422), and the instance settings a request may
leave out (`Yue2Engine.settings`, shown with their explanations on the page).
"""
from __future__ import annotations

import contextlib
import json
import re
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from http import HTTPStatus
from io import BytesIO
from pathlib import Path

from ..request_fields import (
    MAX_LYRICS_CHARS,
    MAX_PROMPT_CHARS,
    MAX_SEED,
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
# The studio's answer to a request it refuses as written.
STUDIO_REFUSED = HTTPStatus.BAD_REQUEST

# -- the window call's shape ------------------------------------------------

# The caller's name for one window call, used to cancel it (`DELETE .../{request_id}`).
REQUEST_ID = re.compile(r"[A-Za-z0-9._-]{1,64}")
# A generation is a YuE2 Studio output folder; its name follows the studio's own rule.
GENERATION_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,179}")
# YuE2's semantic tokens per second: a window's seconds are rounded to these frames.
FPS = 25
WINDOW_FIELDS = {"model", "request_id", "source", "window", "lyrics", "candidates",
                 "context_s", "edge_s", "crossfade_s", "sampling"}
SAMPLING_KEYS = ("temperature", "top_p", "top_k", "repetition_penalty", "penalty_window")
INTEGER_SAMPLING = ("top_k", "penalty_window")
# What a window job hands to the studio, besides the candidate's seed.
STUDIO_WINDOW_FIELDS = ("source", "start_s", "end_s", "lyrics", "context_s", "edge_s",
                        "crossfade_s", "swap_margin_s", "sampling")
# A window's seconds, and its context, are held to these ranges whatever the settings say.
LONGEST_SONG_S = 3600.0
LONGEST_CONTEXT_S = 30.0
LONGEST_RAMP_S = 5.0

# Asks the studio one thing: (method, path, body) -> its JSON answer.
StudioCall = Callable[[str, str, "dict | None"], dict]


class WindowRefusedError(ValueError):
    """A window request that cannot be served as it is: answered 422 with the reason."""


class WindowActiveError(RuntimeError):
    """A window call with this `request_id` is already waiting or running: answered 409."""


class StudioRefusedError(RuntimeError):
    """YuE2 Studio answered with an error; `status` is its HTTP status."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(f"YuE2 Studio rejected the request: {message}")
        self.status = status
        self.reason = message


@dataclass(frozen=True)
class WindowSettings:
    """The instance's window settings (`window_*` in `Yue2Engine.settings`)."""

    context_s: float
    edge_s: float
    crossfade_s: float
    swap_margin_s: float
    max_candidates: int
    min_window_s: float
    max_window_s: float
    sampling: dict

    @classmethod
    def from_params(cls, params: dict) -> WindowSettings:
        """From the validated instance settings, keyed `window_<name>`."""
        return cls(context_s=float(params["window_context_s"]),
                   edge_s=float(params["window_edge_s"]),
                   crossfade_s=float(params["window_crossfade_s"]),
                   swap_margin_s=float(params["window_swap_margin_s"]),
                   max_candidates=int(params["window_max_candidates"]),
                   min_window_s=float(params["window_min_s"]),
                   max_window_s=float(params["window_max_s"]),
                   sampling={key: params[f"window_{key}"] for key in SAMPLING_KEYS})


def was_truncated(value) -> bool:
    """YuE2 reports truncation per stage, as {"abc": False, "semantic": False}.

    A non-empty dictionary is truthy in Python, so `bool()` of that report
    said "truncated" for every song, including ones that finished cleanly.
    A song is truncated only when some stage says so.
    """
    if isinstance(value, dict):
        return any(bool(stage) for stage in value.values())
    return bool(value)


def check_request_id(value) -> str:
    """The caller's `request_id`, refused unless it is 1-64 of [A-Za-z0-9._-]."""
    if not isinstance(value, str) or not REQUEST_ID.fullmatch(value):
        raise WindowRefusedError("request_id must be 1-64 characters of A-Z a-z 0-9 . _ -")
    return value


def check_generation_id(value) -> str:
    """A generation's id (a studio output folder's name), refused when it could leave that folder."""
    if not isinstance(value, str) or not GENERATION_ID.fullmatch(value):
        raise ValueError("generation_id is not a generation's name")
    return value


def _number(body: dict, key: str, default: float, highest: float) -> float:
    value = body.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= highest:
        raise WindowRefusedError(f"{key} must be a number from 0 to {highest}")
    return float(value)


def _sampling(given, defaults: dict) -> dict:
    if given is None:
        return dict(defaults)
    if not isinstance(given, dict):
        raise WindowRefusedError("sampling must be an object")
    unknown = set(given) - set(SAMPLING_KEYS)
    if unknown:
        raise WindowRefusedError("Unknown sampling fields: " + ", ".join(sorted(unknown)))
    merged = {**defaults, **given}
    for key in SAMPLING_KEYS:
        value = merged[key]
        wrong = (type(value) is not int) if key in INTEGER_SAMPLING else (
            isinstance(value, bool) or not isinstance(value, (int, float)))
        if wrong:
            raise WindowRefusedError(f"sampling.{key} must be a number")
    return merged


def _candidates(value, most: int) -> list[int]:
    if not isinstance(value, list) or not 1 <= len(value) <= most:
        raise WindowRefusedError(f"candidates must be a list of 1 to {most} items")
    seeds = []
    for item in value:
        candidate_seed = item.get("seed") if isinstance(item, dict) else None
        if not isinstance(item, dict) or set(item) != {"seed"} or type(candidate_seed) is not int \
                or not 0 <= candidate_seed <= MAX_SEED:
            raise WindowRefusedError("each candidate is {\"seed\": <32-bit unsigned integer>}")
        seeds.append(candidate_seed)
    if len(set(seeds)) != len(seeds):
        raise WindowRefusedError("candidate seeds must differ")
    return seeds


def _span(window, settings: WindowSettings) -> tuple[float, float]:
    """The window's start and end, rounded to frames, within the allowed length."""
    if not isinstance(window, dict) or set(window) != {"start_s", "end_s"}:
        raise WindowRefusedError("window must be {\"start_s\": ..., \"end_s\": ...}")
    start = round(_number(window, "start_s", 0.0, LONGEST_SONG_S) * FPS) / FPS
    end = round(_number(window, "end_s", 0.0, LONGEST_SONG_S) * FPS) / FPS
    if not settings.min_window_s <= end - start <= settings.max_window_s:
        raise WindowRefusedError(f"the window must last {settings.min_window_s}-"
                                 f"{settings.max_window_s} s; it lasts {end - start:.2f} s")
    return start, end


def _source(value) -> str:
    try:
        return check_generation_id(value)
    except ValueError:
        raise WindowRefusedError("source must be the generation_id of an earlier song") from None


def _lyrics(value) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > MAX_LYRICS_CHARS:
        raise WindowRefusedError(f"lyrics must contain 1-{MAX_LYRICS_CHARS} characters")
    return value


def window_request(body: dict, settings: WindowSettings, model_name: str) -> dict:
    """The checked window request, with the instance's settings for what it leaves out."""
    unknown = set(body) - WINDOW_FIELDS
    if unknown:
        raise WindowRefusedError("Unknown YuE2 window fields: " + ", ".join(sorted(unknown)))
    if body.get("model") not in (None, model_name):
        raise WindowRefusedError(f"this engine serves {model_name}")
    start, end = _span(body.get("window"), settings)
    context = _number(body, "context_s", settings.context_s, LONGEST_CONTEXT_S)
    if context < settings.swap_margin_s:
        raise WindowRefusedError(f"context_s must be at least the swap margin "
                                 f"({settings.swap_margin_s} s)")
    return {"request_id": check_request_id(body.get("request_id")),
            "source": _source(body.get("source")), "start_s": start, "end_s": end,
            "lyrics": _lyrics(body.get("lyrics")),
            "seeds": _candidates(body.get("candidates"), settings.max_candidates),
            "context_s": context,
            "edge_s": _number(body, "edge_s", settings.edge_s, LONGEST_RAMP_S),
            "crossfade_s": _number(body, "crossfade_s", settings.crossfade_s, LONGEST_RAMP_S),
            "swap_margin_s": settings.swap_margin_s,
            "sampling": _sampling(body.get("sampling"), settings.sampling)}


def ask_studio(ui_url: str) -> StudioCall:
    """The studio at `ui_url`, asked over HTTP; its refusals as `StudioRefusedError`."""
    def call(method: str, path: str, body: dict | None = None) -> dict:
        payload = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(
            ui_url.rstrip("/") + path, data=payload, method=method,
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=CALL_S) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            message = error.read().decode(errors="replace")
            with contextlib.suppress(ValueError, AttributeError):
                message = json.loads(message).get("detail", message)
            raise StudioRefusedError(error.code, str(message)) from error
    return call


class _WindowCall:
    """One window call in progress: its running studio job, and whether it was cancelled."""

    def __init__(self) -> None:
        self.job_id: int | None = None
        self.cancelled = False


class Yue2WebBackend:
    """YuE2, through the ds-yue-webui studio that keeps the model loaded.

    A request becomes one job in the studio; this waits for it and returns the
    song, the studio's ABC score, and whether a stage was cut short. A window
    request becomes one studio `window` job per candidate, run in order.
    `studio` is how the studio is asked (`ask_studio(ui_url)` unless a test says).
    """

    def __init__(self, ui_url: str, model_name: str, cot: str,
                 window: WindowSettings | None = None, studio: StudioCall | None = None) -> None:
        self.ui_url = ui_url.rstrip("/")
        self.model_name = model_name
        self.cot = cot
        self.window = window
        self._studio = studio or ask_studio(self.ui_url)
        self._calls: dict[str, _WindowCall] = {}
        self._calls_lock = threading.Lock()

    def generate(self, body: dict) -> dict:
        """One song for the request, as base64 WAV, with its score and its generation id."""
        request = self._validate(body)
        params = {"style": request["prompt"], "lyrics": request["lyrics"],
                  "cot": self.cot, "seed": request["seed"]}
        if request["abc"]:
            params["abc"] = request["abc"]
        job = self._wait(self._submit("generate", params, "AI-Lab generation"))
        if job["status"] != "done":
            raise RuntimeError(job.get("error") or f"YuE2 job {job['status']}")
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
            score_abc=score.read_text() if score.is_file() else "",
            generation_id=output.name)

    def generate_window(self, body: dict) -> dict:
        """The source song with one window regenerated, once per candidate seed."""
        if self.window is None:
            raise WindowRefusedError("this YuE2 instance was started without window settings")
        request = window_request(body, self.window, self.model_name)
        call = self._register(request["request_id"])
        try:
            candidates = []
            for candidate_seed in request["seeds"]:
                if call.cancelled:
                    break
                job = self._window_job(request, candidate_seed, call)
                if job["status"] != "done":
                    self._discard(job)
                if job["status"] == "cancelled":
                    break
                if job["status"] != "done":
                    raise RuntimeError(job.get("error") or "YuE2 window job failed")
                candidates.append(_candidate(job))
            return _window_answer(request, candidates, call.cancelled)
        finally:
            with self._calls_lock:
                self._calls.pop(request["request_id"], None)

    def cancel_window(self, request_id: str) -> dict:
        """Stop a window call: no further candidate starts, the running one is cancelled."""
        with self._calls_lock:
            call = self._calls.get(request_id)
            if call is None:
                raise KeyError(f"no window call {request_id} is running")
            call.cancelled = True
            job_id = call.job_id
        if job_id is not None:
            with contextlib.suppress(StudioRefusedError):   # it finished in the meantime
                self._studio("DELETE", f"/api/jobs/{job_id}", None)
        return {"request_id": request_id, "state": "cancelling", "job": job_id}

    def _discard(self, job: dict) -> None:
        """Remove what a candidate that did not finish left in the studio's library."""
        name = Path(job.get("output_dir") or "").name
        if name:
            with contextlib.suppress(StudioRefusedError):    # nothing was written
                self._studio("DELETE", f"/api/library/songs/{name}", None)

    def _register(self, request_id: str) -> _WindowCall:
        with self._calls_lock:
            if request_id in self._calls:
                raise WindowActiveError(f"a window call {request_id} is already running")
            call = self._calls[request_id] = _WindowCall()
        return call

    def _window_job(self, request: dict, candidate_seed: int, call: _WindowCall) -> dict:
        params = {key: request[key] for key in STUDIO_WINDOW_FIELDS}
        params["seed"] = candidate_seed
        try:
            job_id = self._submit("window", params, f"AI-Lab window {request['request_id']}")
        except StudioRefusedError as error:
            if error.status == STUDIO_REFUSED:
                raise WindowRefusedError(error.reason) from None
            raise
        with self._calls_lock:
            call.job_id = job_id
        if call.cancelled:
            self.cancel_window(request["request_id"])
        return self._wait(job_id)

    def _submit(self, kind: str, params: dict, name: str) -> int:
        response = self._studio("POST", "/api/jobs", {"kind": kind, "name": name, "params": params})
        return response["job"]["id"]

    def _wait(self, job_id: int) -> dict:
        """The job once it has ended (done, failed or cancelled)."""
        deadline = time.monotonic() + SONG_LIMIT_S
        while time.monotonic() < deadline:
            job = self._studio("GET", f"/api/jobs/{job_id}", None)
            if job["status"] in {"done", "failed", "cancelled"}:
                return job
            time.sleep(POLL_S)
        raise TimeoutError("YuE2 generation exceeded one hour")

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


def _candidate(job: dict) -> dict:
    """One finished window job as the answer lists it."""
    result = job.get("result") or {}
    window = result.get("window") or {}
    return {"seed": window.get("seed"), "generation_id": Path(job["output_dir"]).name,
            "tokens_changed_pct": window.get("tokens_changed_pct"),
            "short_window": window.get("short_window"),
            "seconds": result.get("audio_seconds"), "timings": window.get("timings"),
            "window_s": window.get("window_s"), "clip_s": window.get("clip_s"),
            "noise_seed": window.get("noise_seed")}


def _window_answer(request: dict, candidates: list[dict], cancelled: bool) -> dict:
    first = candidates[0] if candidates else {}
    return {"request_id": request["request_id"], "source": request["source"],
            "window_s": first.get("window_s") or [request["start_s"], request["end_s"]],
            "clip_s": first.get("clip_s"), "noise_seed": first.get("noise_seed"),
            "candidates": [{key: value for key, value in item.items()
                            if key not in ("window_s", "clip_s", "noise_seed")}
                           for item in candidates],
            "cancelled": cancelled}
