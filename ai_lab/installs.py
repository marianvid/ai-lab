"""Configured package installations and their update watcher.

Environment installation and rollback live in package_install.py.
See docs/package-installs.md for the operating contract.
"""

from __future__ import annotations

import threading
from pathlib import Path

from .events import EventBus
from .gitapps import GitApplicationInstall
from .package_install import CURRENT, PackageInstall

CHECK_INTERVAL_S = 3600.0
STARTUP_DELAY_S = 25.0


class Installs:
    """The package-installed engines this machine knows about, keyed by engine."""

    def __init__(self, settings: dict, bus: EventBus) -> None:
        self._installs: dict[str, PackageInstall] = {}
        self._timer: threading.Thread | None = None
        self._stop = threading.Event()
        for engine_id, engine_settings in (settings or {}).items():
            source = (engine_settings or {}).get("source") or {}
            if source.get("kind") == "git-app":
                self._installs[engine_id] = GitApplicationInstall(engine_id, engine_settings, bus)
                continue
            package = source.get("package")
            root = source.get("root") or _root_of(engine_settings.get("binary", ""))
            if package and root:
                self._installs[engine_id] = PackageInstall(
                    engine_id, root, package, bus,
                    uv=source.get("uv", "uv"), python=source.get("python", ""),
                    install=source.get("install", ""),
                    modules=source.get("modules"),
                    requirements=source.get("requirements"),
                    requires_cuda=bool(source.get("requires_cuda", False)),
                    pip_args=source.get("pip_args"),
                    minimum_versions=source.get("minimum_versions"))

    def all(self) -> list[dict]:
        return [item.status() for item in self._installs.values()]

    def watch(self, interval_s: float = CHECK_INTERVAL_S) -> None:
        """Ask the index for new versions on a timer, in the background.

        So the settings screen is already right when it is opened rather than
        only after somebody presses something — which is why there is no button
        here that only asks a question. A failure is ignored: being offline is
        not worth a message on a page nobody is looking at.
        """
        if self._timer is not None or not self._installs:
            return
        self._timer = threading.Thread(target=self._loop, args=(interval_s,),
                                       daemon=True,
                                       name="ai-lab-install-check")
        self._timer.start()

    def stop(self) -> None:
        self._stop.set()

    def _loop(self, interval_s: float) -> None:
        # A moment's delay first, so starting up is not held back by a network
        # call nobody asked for yet.
        while not self._stop.wait(STARTUP_DELAY_S):
            for install in self._installs.values():
                if self._stop.is_set():
                    return
                try:
                    install.check()
                except Exception:
                    pass
            if self._stop.wait(interval_s):
                return

    def get(self, engine_id: str):
        install = self._installs.get(engine_id)
        if install is None:
            raise KeyError(f"{engine_id} is not installed as packages")
        return install

    def __contains__(self, engine_id: str) -> bool:
        return engine_id in self._installs



def _root_of(binary: str) -> str:
    """…/current/bin/vllm  ->  …  (the directory holding the environments)"""
    path = Path(binary)
    if path.parent.name == "bin":
        return str(path.parent.parent.parent)
    return ""
