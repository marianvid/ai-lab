"""Reclaimable files that are not models.

The browser never sends a filesystem path to delete.  It sends an id from the
configured allow-list, and this service resolves that id to its fixed path.
Model repositories are deliberately absent: their lifecycle belongs to the
Library page.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .disk import tree_size
from .events import EventBus
from .types import ChangeEvent

# Below this depth a configured path is too broad to delete: `/opt/ai` is
# three parts, `/opt/ai/cache` four.
SAFE_DEPTH = 4


@dataclass(frozen=True)
class Guarded:
    """What no reclaimable path may be, hold or sit in (audit m16).

    `trees`: the model roots and repositories, refused as the path, above it or
    inside it. `places`: AI-Lab's state folder, refused as the path or above it
    (its caches inside may be emptied).
    """

    trees: tuple[Path, ...] = ()
    places: tuple[Path, ...] = ()


def guarded_paths(config, state_dir: Path) -> Guarded:
    """The model roots, the repositories and the state folder of this `config`."""
    trees = [Path(item.path) for item in (*config.model_roots, *config.repositories) if item.path]
    return Guarded(trees=tuple(trees), places=(Path(state_dir),))


class Storage:
    """The configured reclaimable places: what each holds, and emptying one."""

    def __init__(self, settings: dict | None, bus: EventBus | None = None,
                 guarded: Callable[[], Guarded] = Guarded) -> None:
        self.bus = bus
        self._guarded = guarded
        self._items: dict[str, dict] = {}
        for raw in (settings or {}).get("reclaimable", []):
            item = dict(raw)
            item_id = str(item.get("id", ""))
            path = Path(str(item.get("path", "")))
            if not item_id or item_id in self._items:
                raise ValueError("Every reclaimable storage item needs a unique id")
            _safe(path, guarded())
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
        _safe(path, self._guarded())
        if path.is_symlink() or path.is_file():
            path.unlink(missing_ok=True)
        elif path.is_dir():
            shutil.rmtree(path)
        if self.bus:
            self.bus.publish(ChangeEvent(topic="storage"))
        return self.view()


def _safe(path: Path, guarded: Guarded) -> None:
    """Refuse broad targets even when a configuration has a typo.

    Depth alone was sized for Linux: on the Mac /Users/<name>/Documents or
    /Volumes/<disk>/work pass it. So the home folder and its direct children,
    the model roots and repositories and the state folder are refused by name.
    """
    if not path.is_absolute() or len(path.parts) < SAFE_DEPTH:
        raise ValueError(f"Unsafe reclaimable storage path: {path}")
    target = path.resolve()
    home = Path.home().resolve()
    if target in (home, *home.parents) or target.parent == home:
        raise ValueError(f"Unsafe reclaimable storage path: {path} is the home folder or one of its own")
    _refuse_guarded(path, target, guarded)


def _refuse_guarded(path: Path, target: Path, guarded: Guarded) -> None:
    """Refuse a path that is, holds or sits in models, or holds the state folder."""
    for tree in guarded.trees:
        tree = tree.resolve()
        if target == tree or target in tree.parents or tree in target.parents:
            raise ValueError(f"Unsafe reclaimable storage path: {path} holds or is inside models ({tree})")
    for place in guarded.places:
        place = place.resolve()
        if target == place or target in place.parents:
            raise ValueError(f"Unsafe reclaimable storage path: {path} holds AI-Lab's state ({place})")
