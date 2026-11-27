"""Configured package installations and their update watcher.

Environment installation and rollback live in package_install.py (Python
packages) and gitapps.py (Git applications such as ComfyUI).
See docs/package-installs.md for the operating contract.
"""

from __future__ import annotations

from pathlib import Path

from ..config.policy import BuildLimits
from ..events import EventBus
from .gitapps import GitApplicationInstall
from .package_install import PackageInstall, PackageOptions
from .update_watch import UpdateWatcher

Install = PackageInstall | GitApplicationInstall


class Installs:
    """The package-installed engines this machine knows about, keyed by engine."""

    def __init__(self, settings: dict, bus: EventBus,
                 limits: BuildLimits | None = None) -> None:
        # An install command's time and search path, and the timer: `limits.builds`.
        limits = limits or BuildLimits()
        found = ((engine_id, _install_for(engine_id, engine_settings or {}, bus, limits))
                 for engine_id, engine_settings in (settings or {}).items())
        self._installs: dict[str, Install] = {
            engine_id: install for engine_id, install in found if install is not None}
        self.watcher = UpdateWatcher(self._installs.values, "ai-lab-install-check",
                                     limits.check_every_s, limits.installs_first_check_s)

    def all(self) -> list[dict]:
        """Every install's status, for the settings page."""
        return [item.status() for item in self._installs.values()]

    def watch(self, interval_s: float | None = None) -> None:
        """Ask the index for new versions on a timer, in the background (`update_watch`).

        So the settings screen is already right when it is opened — which is
        why there is no button here that only asks a question. Every
        `interval_s`, by default `limits.builds.check_every_s`.
        """
        self.watcher.start(interval_s)

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


def _install_for(engine_id: str, engine_settings: dict, bus: EventBus,
                 limits: BuildLimits) -> Install | None:
    """The install an engine's settings describe, or None when they describe none."""
    source = engine_settings.get("source") or {}
    if source.get("kind") == "git-app":
        return GitApplicationInstall(engine_id, engine_settings, bus, limits=limits)
    package = source.get("package")
    root = source.get("root") or _root_of(engine_settings.get("binary", ""))
    if not (package and root):
        return None
    options = PackageOptions.from_source(source, timeout_s=limits.install_s,
                                         system_path=limits.system_path)
    return PackageInstall(engine_id, root, package, bus, options)


def _root_of(binary: str) -> str:
    """…/current/bin/vllm  ->  …  (the directory holding the environments)"""
    path = Path(binary)
    if path.parent.name == "bin":
        return str(path.parent.parent.parent)
    return ""
