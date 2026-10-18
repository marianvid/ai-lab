"""Reclaimable files that are not models.

The browser never sends a filesystem path to delete.  It sends an id from the
configured allow-list, and this service resolves that id to its fixed path.
Model repositories are deliberately absent: their lifecycle belongs to the
Library page.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from .disk import tree_size
from .events import EventBus
from .types import ChangeEvent

# Below this depth a configured path is too broad to delete: `/opt/ai` is
# three parts, `/opt/ai/cache` four.
SAFE_DEPTH = 4


class Storage:
    """The configured reclaimable places: what each holds, and emptying one."""

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
        """What is there to reclaim now, with sizes."""
        items = []
        for configured in self._items.values():
            path = configured["path"]
            exists = path.exists() or path.is_symlink()
            size = tree_size(path)
            # Configuration describes places worth knowing about; the page is
            # an inventory of what is actually there. A missing leftover, or
            # an empty cache directory recreated by a runtime, offers nothing
            # to reclaim and should not leave a permanent empty row behind.
            if not exists or (configured["kind"] == "cache" and size == 0):
                continue
            items.append({
                **{key: value for key, value in configured.items() if key != "path"},
                "path": str(path),
                "exists": True,
                "size_bytes": size,
            })
        return {"items": items,
                "recoverable_bytes": sum(item["size_bytes"] for item in items)}

    def clear(self, item_id: str) -> dict:
        """Delete what one configured item holds, by its id."""
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
    if not path.is_absolute() or len(path.parts) < SAFE_DEPTH:
        raise ValueError(f"Unsafe reclaimable storage path: {path}")
