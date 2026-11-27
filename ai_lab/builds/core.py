"""Configured source checkouts and their update watcher.

The versioned build and rollback implementation is in source_build.py.
See docs/source-builds.md for the operating contract.
"""

from __future__ import annotations

from ..config.policy import BuildLimits
from ..events import EventBus
from .source_build import BuildOptions, SourceBuild
from .update_watch import UpdateWatcher


class Builds:
    """The source checkouts this installation knows about, keyed by engine."""

    def __init__(self, settings: dict, bus: EventBus,
                 limits: BuildLimits | None = None) -> None:
        limits = limits or BuildLimits()
        self._builds: dict[str, SourceBuild] = {}
        self.watcher = UpdateWatcher(self._builds.values, "ai-lab-version-check",
                                     limits.check_every_s, limits.builds_first_check_s)
        for engine_id, engine_settings in (settings or {}).items():
            source = (engine_settings or {}).get("source")
            if source and source.get("path"):
                self._builds[engine_id] = SourceBuild(
                    engine_id, source["path"], bus,
                    BuildOptions.from_source(source, timeout_s=limits.build_s))

    def all(self, with_sizes: bool = True) -> list[dict]:
        """Every checkout's status, for the settings page."""
        return [item.status(with_sizes=with_sizes) for item in self._builds.values()]

    def get(self, engine_id: str) -> SourceBuild:
        """The checkout of one engine; KeyError when it has none configured."""
        build = self._builds.get(engine_id)
        if build is None:
            raise KeyError(f"No source build configured for {engine_id}")
        return build

    def watch(self, interval_s: float | None = None) -> None:
        """Ask upstream for new versions on a timer, in the background (`update_watch`).

        Every `interval_s`, by default `limits.builds.check_every_s`.
        """
        self.watcher.start(interval_s)

    def stop(self) -> None:
        """Stop the timer."""
        self.watcher.stop()
