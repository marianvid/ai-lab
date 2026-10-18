"""Configured package installations and their update watcher.

Environment installation and rollback live in package_install.py (Python
packages) and gitapps.py (Git applications such as ComfyUI).
See docs/package-installs.md for the operating contract.
"""

from __future__ import annotations

from pathlib import Path

from .events import EventBus
from .gitapps import GitApplicationInstall
from .package_install import PackageInstall, PackageOptions
from .update_watch import UpdateWatcher

CHECK_INTERVAL_S = 3600.0
STARTUP_DELAY_S = 25.0

Install = PackageInstall | GitApplicationInstall


class Installs:
    """The package-installed engines this machine knows about, keyed by engine."""

    def __init__(self, settings: dict, bus: EventBus) -> None:
        self._installs: dict[str, Install] = {}
        self.watcher = UpdateWatcher(self._installs.values, "ai-lab-install-check")
        for engine_id, engine_settings in (settings or {}).items():
            install = _install_for(engine_id, engine_settings or {}, bus)
            if install is not None:
                self._installs[engine_id] = install

    def all(self) -> list[dict]:
        """Every install's status, for the settings page."""
        return [item.status() for item in self._installs.values()]

    def watch(self, interval_s: float = CHECK_INTERVAL_S) -> None:
        """Ask the index for new versions on a timer, in the background (`update_watch`).

        So the settings screen is already right when it is opened — which is
        why there is no button here that only asks a question.
        """
        self.watcher.start(interval_s, STARTUP_DELAY_S)

    def stop(self) -> None:
        """Stop the timer."""
        self.watcher.stop()

    def get(self, engine_id: str) -> Install:
        """The install of one engine; KeyError when it is not installed as packages."""
        install = self._installs.get(engine_id)
        if install is None:
            raise KeyError(f"{engine_id} is not installed as packages")
        return install

    def __contains__(self, engine_id: object) -> bool:
        return engine_id in self._installs


def _install_for(engine_id: str, engine_settings: dict, bus: EventBus) -> Install | None:
    """The install an engine's settings describe, or None when they describe none."""
    source = engine_settings.get("source") or {}
    if source.get("kind") == "git-app":
        return GitApplicationInstall(engine_id, engine_settings, bus)
    package = source.get("package")
    root = source.get("root") or _root_of(engine_settings.get("binary", ""))
    if not (package and root):
        return None
    return PackageInstall(engine_id, root, package, bus, PackageOptions.from_source(source))


def _root_of(binary: str) -> str:
    """…/current/bin/vllm  ->  …  (the directory holding the environments)"""
    path = Path(binary)
    if path.parent.name == "bin":
        return str(path.parent.parent.parent)
    return ""
