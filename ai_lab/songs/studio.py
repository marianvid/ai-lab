"""A YuE2 entry's songs in its engine's folder, and what YuE2 Studio decides about them.

The decisions the routes used to make (L61): which Studio jobs make a song busy,
and whether a song is removed through the Studio or straight from the folder.
`api/routes/music_windows.py` only reads the request and hands it here; the
window calls in flight come in as `editing`, a question about a generation.
"""
from __future__ import annotations

import json
import shutil
import urllib.error
import urllib.request
from collections.abc import Callable
from http import HTTPStatus
from pathlib import Path
from urllib.parse import quote

from ..engines.base import MUSIC_WINDOW_PATHS
from ..gateway.errors import ModelNotLoadedError, WindowCallActiveError
from ..network import LOOPBACK
from ..shared.song_names import check_generation_id

WINDOW_PATH = MUSIC_WINDOW_PATHS[0]
# How long YuE2 Studio may take to answer one question (its jobs, a delete).
STUDIO_CALL_S = 30
# How many of the Studio's latest jobs are looked at for one still working on a song;
# the Studio runs one job at a time, so the pending and running ones are among the latest.
STUDIO_JOBS_LOOKED_AT = 50
# How YuE2 Studio's refusal to delete a song a job is still writing begins (`library.delete_directory`).
STUDIO_IN_USE = "an active job is writing"


class SongFolders:
    """A YuE2 entry's songs, read and removed in its engine's folder (`transfers.Songs`)."""

    def __init__(self, gateway, operations, editing: Callable[[str], str | None]) -> None:
        self.gateway = gateway
        self.operations = operations
        self.editing = editing

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
        return self.engine(query).editor_port(port)

    def refuse_busy(self, query: dict | None, generation_id: str) -> None:
        """409 when a window call edits this generation, or a Studio job writes or reads it."""
        self._refuse_edited(generation_id)
        port = self.studio_port(query)
        if port is not None and studio_uses(port, generation_id):
            raise WindowCallActiveError(f"a YuE2 Studio job is working on {generation_id}")

    def delete(self, query: dict | None, generation_id: str) -> dict:
        """Remove the generation (a candidate not kept).

        Through YuE2 Studio when the engine is up, so its own guard refuses a folder
        a job is writing and its cached mp3 goes too; straight from the folder when it
        is down, since then nothing can be writing. The source of a window call that
        is waiting or running is refused (409) either way.
        """
        folder = self.folder(query, generation_id)
        self._refuse_edited(generation_id)
        port = self.studio_port(query)
        if port is None:
            shutil.rmtree(folder)
        else:
            studio_delete(port, generation_id)
        return {"deleted": generation_id}

    def _refuse_edited(self, generation_id: str) -> None:
        editing = self.editing(generation_id)
        if editing is not None:
            raise WindowCallActiveError(f"{generation_id} is the source of window call {editing}")


def studio_delete(studio_port: int, generation_id: str) -> None:
    """YuE2 Studio's own delete of a library song; its refusal of a folder in use is a 409.

    The Studio answers both refusals with 404 and says which in `detail`
    (`routes.delete_song`): an unknown song, or "an active job is writing to this
    directory" (`library.delete_directory`), which is not "not found".
    """
    request = urllib.request.Request(
        f"http://{LOOPBACK}:{studio_port}/api/library/songs/{quote(generation_id, safe='')}",
        method="DELETE")
    try:
        with urllib.request.urlopen(request, timeout=STUDIO_CALL_S):
            return
    except urllib.error.HTTPError as error:
        detail = _detail(error)
        if error.code == HTTPStatus.NOT_FOUND and detail.startswith(STUDIO_IN_USE):
            raise WindowCallActiveError(f"YuE2 Studio refused to delete {generation_id}: {detail}") from error
        if error.code == HTTPStatus.NOT_FOUND:
            raise FileNotFoundError(f"no generation {generation_id}") from error
        raise RuntimeError(f"YuE2 Studio could not delete {generation_id}: HTTP {error.code}") from error


def studio_uses(studio_port: int, generation_id: str) -> bool:
    """Whether a pending or running Studio job writes this generation, or edits it as its source."""
    url = f"http://{LOOPBACK}:{studio_port}/api/jobs?limit={STUDIO_JOBS_LOOKED_AT}"
    with urllib.request.urlopen(url, timeout=STUDIO_CALL_S) as response:
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
