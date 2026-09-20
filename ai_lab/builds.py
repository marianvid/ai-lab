"""Configured source checkouts and their update watcher.

The versioned build and rollback implementation is in source_build.py.
See docs/source-builds.md for the operating contract.
"""

from __future__ import annotations

import threading

from .events import EventBus
from .source_build import SourceBuild
from .source_versions import DEFAULT_LINE, Version, _mark, _marked

CHECK_INTERVAL_S = 3600.0
STARTUP_DELAY_S = 20.0


class Builds:
    """The source checkouts this installation knows about, keyed by engine."""

    def __init__(self, settings: dict, bus: EventBus) -> None:
        self._builds: dict[str, SourceBuild] = {}
        self._timer: threading.Thread | None = None
        self._stop = threading.Event()
        for engine_id, engine_settings in (settings or {}).items():
            source = (engine_settings or {}).get("source")
            if source and source.get("path"):
                self._builds[engine_id] = SourceBuild(
                    engine_id, source["path"], bus, jobs=source.get("jobs"),
                    line=source.get("line", DEFAULT_LINE),
                    builds=source.get("builds", ""),
                    legacy_build=source.get("legacy_build", ""),
                    cmake_args=source.get("cmake_args"))

    def all(self, with_sizes: bool = True) -> list[dict]:
        return [item.status(with_sizes=with_sizes) for item in self._builds.values()]

    def get(self, engine_id: str) -> SourceBuild:
        build = self._builds.get(engine_id)
        if build is None:
            raise KeyError(f"No source build configured for {engine_id}")
        return build

    def watch(self, interval_s: float = CHECK_INTERVAL_S) -> None:
        """Ask upstream for new versions on a timer, in the background.

        So that the settings screen is already right when it is opened, rather
        than only after someone presses a button. A failure is ignored: being
        offline, or upstream being unreachable, is not worth a message on a
        page nobody is looking at.
        """
        if self._timer is not None or not self._builds:
            return
        self._timer = threading.Thread(target=self._loop, args=(interval_s,),
                                       daemon=True, name="ai-lab-version-check")
        self._timer.start()

    def stop(self) -> None:
        self._stop.set()

    def _loop(self, interval_s: float) -> None:
        # A moment's delay first, so starting up is not held back by a network
        # call nobody asked for yet.
        while not self._stop.wait(STARTUP_DELAY_S):
            for build in self._builds.values():
                if self._stop.is_set():
                    return
                try:
                    build.check()
                except Exception:
                    pass
            if self._stop.wait(interval_s):
                return
