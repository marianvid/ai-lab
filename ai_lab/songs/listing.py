"""This machine's songs, newest first, with what recognises each one. A read: nothing changes."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .transfers import Songs

# What a list entry quotes from the song's request, at most (characters).
QUOTED_CHARS = 300


def generations(files: Songs, query: dict | None) -> dict:
    """This machine's generations, newest first, `?limit=` (at most `list_most`) from `?offset=`."""
    engine = files.engine(query)
    most = engine.transfer.list_most
    limit = _count(query, "limit", most, 1, most)
    offset = _count(query, "offset", 0, 0, None)
    root = engine.generations_dir()
    folders = [path for path in root.iterdir()
               if path.is_dir() and not path.name.startswith(".")] if root.is_dir() else []
    # Sorted by time first, so only the page's folders are read in detail. A folder that
    # vanishes meanwhile (deleted while this runs) is left out rather than failing the list.
    dated = sorted(((when, path) for path in folders if (when := _made(path)) is not None),
                   key=lambda item: (item[0], item[1].name), reverse=True)
    page = [entry for _when, path in dated[offset:offset + limit] if (entry := _entry(path)) is not None]
    return {"machine": engine.transfer.machine, "total": len(dated), "offset": offset,
            "limit": limit, "generations": page}


def _made(folder: Path) -> float | None:
    """When the generation was finished (its `result.json`, else the folder); None if it is gone."""
    for path in (folder / "result.json", folder):
        try:
            return path.stat().st_mtime
        except FileNotFoundError:
            continue
    return None


def _count(query: dict | None, key: str, default: int, lowest: int, highest: int | None) -> int:
    raw = (query or {}).get(key)
    if raw is None:
        return default
    if not raw.isdigit() or int(raw) < lowest or (highest is not None and int(raw) > highest):
        top = "" if highest is None else f" to {highest}"
        raise ValueError(f"{key} must be a whole number from {lowest}{top}")
    return int(raw)


def _entry(folder: Path) -> dict | None:
    """One generation: its id, when it was made, its size, and what recognises it; None if it is gone."""
    try:
        made = _made(folder)
        sizes = [path.stat().st_size for path in folder.iterdir()
                 if path.is_file() and not path.name.startswith(".")]
    except FileNotFoundError:
        return None
    if made is None:
        return None
    request = _json_file(folder / "request.json")
    result = _json_file(folder / "result.json")
    sources = result.get("sources") if isinstance(result.get("sources"), dict) else {}
    seconds = result.get("audio_seconds")
    return {
        "generation_id": folder.name,
        "made_at": _iso(made),
        "bytes": sum(sizes),
        "files": len(sizes),
        "complete": result.get("status") == "complete",
        "kind": "window" if sources else "song",
        "source": sources.get("song") if sources else None,
        "seconds": round(float(seconds), 2) if isinstance(seconds, (int, float)) else None,
        "style": _quote(request.get("style")),
        "lyrics_start": _quote(request.get("lyrics")),
        "seed": request.get("seed"),
        "cot": request.get("cot"),
        "cfg_scale": request.get("cfg_scale"),
    }


def _json_file(path: Path) -> dict:
    """A small JSON file of the Studio's as a dictionary; empty when missing or unreadable."""
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _quote(value) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value if len(value) <= QUOTED_CHARS else value[:QUOTED_CHARS] + "…"


def _iso(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat(timespec="seconds")
