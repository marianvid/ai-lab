"""Regenerating a window of a YuE2 song: the calls by `request_id`, and the songs' files.

`POST /v1/audio/music/window-generations` is forwarded like any music call (it waits for the
card, `gateway.py`); what is added here is what the front door must know about it:

- the call's `request_id`, so `DELETE /v1/audio/music/window-generations/{request_id}` can
  stop it: still waiting for the card, it is dropped from the queue; already running, the
  engine is told to cancel it;
- the songs themselves, read from the engine's folder without the model loaded:
  `GET /v1/audio/music/generations/{generation_id}/audio` (the whole song, FLAC),
  `GET .../{generation_id}/window-audio` (the regenerated clip, WAV) and
  `DELETE /v1/audio/music/generations/{generation_id}`. The engine is found by the
  `model` query parameter (`?model=<the YuE2 entry>`).
"""
from __future__ import annotations

import json
import shutil
import threading
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from http import HTTPStatus
from pathlib import Path
from urllib.parse import quote

from ...engines.base import MUSIC_WINDOW_PATHS
from ...engines.isolated import ui_port
from ...gateway.errors import ModelNotLoadedError, WindowCallActiveError, WindowRequestError
from ...music.backends.yue2_web_backend import REQUEST_ID, check_generation_id
from ...network import LOOPBACK
from ..passthrough import Passthrough
from .music_transfers import register_transfers

WINDOW_PATH = MUSIC_WINDOW_PATHS[0]
# How long the engine may take to say it is cancelling.
CANCEL_CALL_S = 30
# A song's file is sent in pieces of this size.
CHUNK_BYTES = 1 << 20
# How many of the Studio's latest jobs are looked at for one still working on a song;
# the Studio runs one job at a time, so the pending and running ones are among the latest.
STUDIO_JOBS_LOOKED_AT = 50
# How YuE2 Studio's refusal to delete a song a job is still writing begins (`library.delete_directory`).
STUDIO_IN_USE = "an active job is writing"
# The files a generation serves, by route: name in its folder, and media type.
FILES = {"audio": ("audio.flac", "audio/flac"), "window-audio": ("window.wav", "audio/wav")}


def window_request_id(payload) -> str:
    """The call's `request_id`, refused (422) unless it is 1-64 of [A-Za-z0-9._-]."""
    value = payload.get("request_id") if isinstance(payload, dict) else None
    if not isinstance(value, str) or not REQUEST_ID.fullmatch(value):
        raise WindowRequestError("a window call needs a request_id of 1-64 characters "
                                 "of A-Z a-z 0-9 . _ -")
    return value


class WindowCalls:
    """Window calls by `request_id`: waiting for the card, or running on an engine."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._calls: dict[str, dict] = {}

    def begin(self, request_id: str, source: str | None = None) -> None:
        """A call starts waiting; a 409 when that id is already in use."""
        with self._lock:
            if request_id in self._calls:
                raise WindowCallActiveError(f"a window call {request_id} is already waiting or running")
            self._calls[request_id] = {"port": None, "cancelled": False, "source": source}

    def in_use(self, generation_id: str) -> str | None:
        """The window call editing this generation, if one is waiting or running."""
        with self._lock:
            return next((request_id for request_id, call in self._calls.items()
                         if call["source"] == generation_id), None)

    def running(self, request_id: str, port: int) -> None:
        """The call has its place and is on the engine at `port`."""
        with self._lock:
            if request_id in self._calls:
                self._calls[request_id]["port"] = port

    def wanted(self, request_id: str) -> bool:
        """False once the call was cancelled while it waited."""
        with self._lock:
            call = self._calls.get(request_id)
            return call is not None and not call["cancelled"]

    def end(self, request_id: str) -> None:
        """The call is over, however it ended."""
        with self._lock:
            self._calls.pop(request_id, None)

    def active(self) -> list[str]:
        """The ids waiting or running, for the page and the tests."""
        with self._lock:
            return sorted(self._calls)

    def cancel(self, request_id: str) -> dict:
        """Stop the call: dropped while it waits, cancelled by the engine while it runs."""
        with self._lock:
            call = self._calls.get(request_id)
            if call is None:
                raise KeyError(f"no window call {request_id} is waiting or running")
            if call["port"] is None:
                call["cancelled"] = True
                return {"request_id": request_id, "state": "cancelled", "was": "waiting"}
            port = call["port"]
        return _engine_cancel(port, request_id)


def _engine_cancel(port: int, request_id: str) -> dict:
    request = urllib.request.Request(
        f"http://{LOOPBACK}:{port}{WINDOW_PATH}/{quote(request_id, safe='')}", method="DELETE")
    try:
        with urllib.request.urlopen(request, timeout=CANCEL_CALL_S) as response:
            answer = json.load(response)
    except urllib.error.HTTPError as error:
        if error.code == HTTPStatus.NOT_FOUND:
            return {"request_id": request_id, "state": "finished", "was": "running"}
        raise RuntimeError(f"the engine could not cancel {request_id}: HTTP {error.code}") from error
    return {**answer, "was": "running"}


@dataclass(frozen=True)
class WindowAsk:
    """What a window call asks the card for: its body, the model, its start settings, its client."""

    payload: dict
    model: str
    settings: dict | None
    alive: Callable[[], bool] | None


def window_call(gateway, path: str, ask: WindowAsk, windows: WindowCalls,
                send: Callable[[object], Passthrough]) -> Passthrough:
    """A window call: known by its `request_id` from the queue until its answer is closed.

    `send(lease)` forwards the request on the place it was given; a failure gives
    the place back.
    """
    request_id = window_request_id(ask.payload)
    source = ask.payload.get("source")
    windows.begin(request_id, source if isinstance(source, str) else None)
    try:
        def still_wanted() -> bool:
            return windows.wanted(request_id) and (ask.alive is None or ask.alive())
        lease = gateway.acquire(ask.model, shape=path, settings=ask.settings,
                                still_wanted=still_wanted)
        windows.running(request_id, lease.port)
        if not windows.wanted(request_id):
            # Cancelled between getting the place and recording it: nothing is sent.
            lease.release()
            windows.end(request_id)
            return _cancelled_before_start(request_id)
        try:
            answer = send(lease)
        except BaseException:
            lease.release()
            raise
    except BaseException:
        windows.end(request_id)
        raise
    return ended_with(answer, lambda: windows.end(request_id))


def _cancelled_before_start(request_id: str) -> Passthrough:
    """The answer of a window call cancelled before it reached the engine."""
    body = {"request_id": request_id, "candidates": [], "cancelled": True}
    return Passthrough(status=HTTPStatus.OK, content_type="application/json",
                       chunks=iter([json.dumps(body).encode()]), headers={})


def ended_with(answer: Passthrough, done) -> Passthrough:
    """The engine's answer, calling `done` once when it is closed."""
    close = answer.close

    def closing() -> None:
        try:
            close()
        finally:
            done()
    answer.close = closing
    return answer


class GenerationFiles:
    """A YuE2 entry's songs, read and removed in its engine's folder."""

    def __init__(self, gateway, operations, windows: WindowCalls | None = None) -> None:
        self.gateway = gateway
        self.operations = operations
        self.windows = windows or WindowCalls()

    def engine(self, query: dict | None):
        """The engine of the entry named by `?model=`, refused unless it keeps generations."""
        model = (query or {}).get("model")
        if not model:
            raise ValueError("name the YuE2 entry with ?model=")
        instance = self.gateway.resolve(model, self.operations.configured())
        engine = self.operations.engine(instance["engine"])
        if getattr(engine, "generations_dir", None) is None:
            raise ValueError(f"{model} keeps no generations")
        return engine

    def folder(self, query: dict | None, generation_id: str) -> Path:
        """The generation's folder, for the entry named by `?model=`."""
        folder = self.engine(query).generations_dir() / check_generation_id(generation_id)
        if not folder.is_dir():
            raise FileNotFoundError(f"no generation {generation_id}")
        return folder

    def studio_port(self, query: dict | None) -> int | None:
        """The YuE2 Studio's port while the entry is loaded, else None."""
        try:
            port, _name = self.gateway.where_loaded((query or {})["model"], WINDOW_PATH)
        except ModelNotLoadedError:
            return None
        return int(ui_port(port))

    def refuse_busy(self, query: dict | None, generation_id: str) -> None:
        """409 when a window call edits this generation, or a Studio job writes or reads it."""
        editing = self.windows.in_use(generation_id)
        if editing is not None:
            raise WindowCallActiveError(f"{generation_id} is the source of window call {editing}")
        port = self.studio_port(query)
        if port is not None and _studio_uses(port, generation_id):
            raise WindowCallActiveError(f"a YuE2 Studio job is working on {generation_id}")

    def send(self, query: dict | None, generation_id: str, what: str) -> Passthrough:
        """One of the generation's files (`FILES`), in pieces."""
        name, media_type = FILES[what]
        path = self.folder(query, generation_id) / name
        if not path.is_file():
            raise FileNotFoundError(f"generation {generation_id} has no {name}")
        return Passthrough(status=200, content_type=media_type, chunks=_pieces(path),
                           headers={"Content-Disposition": f'attachment; filename="{generation_id}-{name}"'})

    def delete(self, query: dict | None, generation_id: str) -> dict:
        """Remove the generation (a candidate not kept).

        Through YuE2 Studio when the engine is up, so its own guard refuses a folder
        a job is writing and its cached mp3 goes too; straight from the folder when it
        is down, since then nothing can be writing. The source of a window call that
        is waiting or running is refused (409) either way.
        """
        folder = self.folder(query, generation_id)
        editing = self.windows.in_use(generation_id)
        if editing is not None:
            raise WindowCallActiveError(f"{generation_id} is the source of window call {editing}")
        port = self.studio_port(query)
        if port is None:
            shutil.rmtree(folder)
        else:
            _studio_delete(port, generation_id)
        return {"deleted": generation_id}


def _studio_delete(studio_port: int, generation_id: str) -> None:
    """YuE2 Studio's own delete of a library song; its refusal of a folder in use is a 409.

    The Studio answers both refusals with 404 and says which in `detail`
    (`routes.delete_song`): an unknown song, or "an active job is writing to this
    directory" (`library.delete_directory`), which is not "not found".
    """
    request = urllib.request.Request(
        f"http://{LOOPBACK}:{studio_port}/api/library/songs/{quote(generation_id, safe='')}",
        method="DELETE")
    try:
        with urllib.request.urlopen(request, timeout=CANCEL_CALL_S):
            return
    except urllib.error.HTTPError as error:
        detail = _detail(error)
        if error.code == HTTPStatus.NOT_FOUND and detail.startswith(STUDIO_IN_USE):
            raise WindowCallActiveError(f"YuE2 Studio refused to delete {generation_id}: {detail}") from error
        if error.code == HTTPStatus.NOT_FOUND:
            raise FileNotFoundError(f"no generation {generation_id}") from error
        raise RuntimeError(f"YuE2 Studio could not delete {generation_id}: HTTP {error.code}") from error


def _studio_uses(studio_port: int, generation_id: str) -> bool:
    """Whether a pending or running Studio job writes this generation, or edits it as its source."""
    url = f"http://{LOOPBACK}:{studio_port}/api/jobs?limit={STUDIO_JOBS_LOOKED_AT}"
    with urllib.request.urlopen(url, timeout=CANCEL_CALL_S) as response:
        jobs = json.load(response).get("jobs", [])
    return any(job.get("status") in ("pending", "running")
               and (Path(job.get("output_dir") or "").name == generation_id
                    or (job.get("params") or {}).get("source") == generation_id)
               for job in jobs)


def _detail(error: urllib.error.HTTPError) -> str:
    """The Studio's `detail` sentence, or the body as it came."""
    body = error.read().decode(errors="replace")
    try:
        detail = json.loads(body).get("detail", body)
    except (ValueError, AttributeError):
        return body
    return str(detail)


def _pieces(path: Path) -> Iterator[bytes]:
    with path.open("rb") as handle:
        while piece := handle.read(CHUNK_BYTES):
            yield piece


def register_files(router, files: GenerationFiles, windows: WindowCalls) -> None:
    """The generation routes (files, bundle, pull) and the window cancel."""
    base = "/v1/audio/music/generations/{generation_id}"
    router.add("GET", base + "/audio",
               lambda generation_id, query=None, **_: files.send(query, generation_id, "audio"))
    router.add("GET", base + "/window-audio",
               lambda generation_id, query=None, **_: files.send(query, generation_id, "window-audio"))
    router.add("DELETE", base,
               lambda generation_id, query=None, **_: files.delete(query, generation_id))
    router.add("DELETE", WINDOW_PATH + "/{request_id}",
               lambda request_id, **_: windows.cancel(request_id))
    register_transfers(router, files)
