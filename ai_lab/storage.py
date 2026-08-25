"""Reclaimable files that are not models.

The browser never sends a filesystem path to delete.  It sends an id from the
configured allow-list, and this service resolves that id to its fixed path.
Model repositories are deliberately absent: their lifecycle belongs to the
Library page.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from .events import EventBus
from .types import ChangeEvent


class Storage:
    def __init__(self, settings: dict | None, bus: EventBus | None = None) -> None:
        self.bus = bus
        self._items: dict[str, dict] = {}
        for raw in (settings or {}).get("reclaimable", []):
            item = dict(raw)
            item_id = str(item.get("id", ""))
            path = Path(str(item.get("path", "")))
            if not item_id or item_id in self._items:
                raise ValueError("Every reclaimable storage item needs a unique id")
            _safe(path)
            self._items[item_id] = {
                "id": item_id,
                "name": str(item.get("name") or item_id),
                "path": path,
                "kind": str(item.get("kind") or "cache"),
                "description": str(item.get("description") or ""),
            }

    def view(self) -> dict:
        items = []
        for configured in self._items.values():
            path = configured["path"]
            size = _size(path)
            items.append({
                **{key: value for key, value in configured.items() if key != "path"},
                "path": str(path),
                "exists": path.exists() or path.is_symlink(),
                "size_bytes": size,
            })
        return {"items": items,
                "recoverable_bytes": sum(item["size_bytes"] for item in items)}

    def clear(self, item_id: str) -> dict:
        try:
            item = self._items[item_id]
        except KeyError:
            raise KeyError(f"No reclaimable storage item called {item_id}") from None
        path = item["path"]
        _safe(path)
        if path.is_symlink() or path.is_file():
            path.unlink(missing_ok=True)
        elif path.is_dir():
            shutil.rmtree(path)
        if self.bus:
            self.bus.publish(ChangeEvent(topic="storage"))
        return self.view()


def _safe(path: Path) -> None:
    """Refuse broad targets even when a configuration has a typo."""
    if not path.is_absolute() or len(path.parts) < 4:
        raise ValueError(f"Unsafe reclaimable storage path: {path}")


def _size(path: Path) -> int:
    if path.is_file() and not path.is_symlink():
        try:
            return path.stat().st_size
        except OSError:
            return 0
    total = 0
    try:
        for item in path.rglob("*"):
            try:
                if item.is_file() and not item.is_symlink():
                    total += item.stat().st_size
            except OSError:
                pass
    except OSError:
        pass
    return total
