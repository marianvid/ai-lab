"""Configured source checkouts and their update watcher.

The versioned build and rollback implementation is in source_build.py.
See docs/source-builds.md for the operating contract.
"""

from __future__ import annotations

from ..events import EventBus
from .source_build import BuildOptions, SourceBuild
from .update_watch import UpdateWatcher

CHECK_INTERVAL_S = 3600.0
STARTUP_DELAY_S = 20.0


class Builds:
    """The source checkouts this installation knows about, keyed by engine."""

    def __init__(self, settings: dict, bus: EventBus) -> None:
        self._builds: dict[str, SourceBuild] = {}
        self.watcher = UpdateWatcher(self._builds.values, "ai-lab-version-check")
        for engine_id, engine_settings in (settings or {}).items():
            source = (engine_settings or {}).get("source")
            if source and source.get("path"):
                self._builds[engine_id] = SourceBuild(
                    engine_id, source["path"], bus, BuildOptions.from_source(source))

    def all(self, with_sizes: bool = True) -> list[dict]:
        """Every checkout's status, for the settings page."""
        return [item.status(with_sizes=with_sizes) for item in self._builds.values()]

    def get(self, engine_id: str) -> SourceBuild:
        """The checkout of one engine; KeyError when it has none configured."""
        build = self._builds.get(engine_id)
        if build is None:
            raise KeyError(f"No source build configured for {engine_id}")
        return build

    def watch(self, interval_s: float = CHECK_INTERVAL_S) -> None:
        """Ask upstream for new versions on a timer, in the background (`update_watch`)."""
        self.watcher.start(interval_s, STARTUP_DELAY_S)

    def stop(self) -> None:
        """Stop the timer."""
        self.watcher.stop()
