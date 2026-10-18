"""The installed versions of one package engine, as folders beside each other.

`EnvironmentFolders` is held by `PackageInstall`. It keeps no state of its own
beyond the folder and the package name (everything else is on disk), so it
needs no lock; `PackageInstall` owns the lock and never installs and switches
at once. Every version lives in its own
`.venv-<version>` folder under `root`; the original install, from before this
scheme, is `.venv`. A `current` link names the one the engine is launched
from, so switching is moving the link: going back is one press and deletes
nothing.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

from ..disk import tree_size

CURRENT = "current"
PREFIX = ".venv-"
NAME = re.compile(r"^\.venv(?:-(.+))?$")
FIRST = ".venv"
# How long asking an environment for its package's version may take.
VERSION_QUESTION_S = 60


class Environment:
    """One installed version, as a folder on disk."""

    __slots__ = ("active", "movable", "name", "path", "size_bytes", "version")

    def __init__(self, path: Path, active: bool = False, version: str = "") -> None:
        self.path = path
        self.name = path.name
        match = NAME.match(path.name)
        self.version = version or (match.group(1) if match and match.group(1)
                                   else path.name)
        self.active = active
        self.size_bytes = 0
        # The one that predates this scheme sits at a fixed path that its own
        # scripts name. Everything else was created where it stands.
        self.movable = path.name != FIRST

    def json(self) -> dict:
        """The row the settings page shows for this environment."""
        return {"name": self.name, "version": self.version,
                "active": self.active, "size_bytes": self.size_bytes,
                "path": str(self.path), "movable": self.movable}


def version_in(environment: Path, package: str) -> str:
    """The version of `package` installed in an environment; "" when it cannot tell."""
    program = f"import importlib.metadata as m; print(m.version({package!r}))"
    try:
        result = subprocess.run([str(environment / "bin" / "python"), "-c", program],
                                capture_output=True, text=True, timeout=VERSION_QUESTION_S)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def replace_link(link: Path, target: Path) -> None:
    """Point `link` at `target` in one step.

    Done by writing a new link beside the old one and renaming it over the
    top, which the filesystem does at once. A link that is deleted and then
    recreated has a moment in between where it points at nothing, and anything
    starting in that moment fails for a reason nobody would guess.
    """
    temporary = link.with_name(f".{link.name}-new")
    if temporary.exists() or temporary.is_symlink():
        temporary.unlink()
    temporary.symlink_to(target, target_is_directory=True)
    temporary.replace(link)


class EnvironmentFolders:
    """The environments under `root`, and the `current` link naming the one in use."""

    def __init__(self, root: Path, package: str) -> None:
        self.root = root
        self.package = package

    @property
    def link(self) -> Path:
        """The `current` link naming the environment in use."""
        return self.root / CURRENT

    def active(self) -> Path | None:
        """The folder the engine is currently launched from."""
        try:
            return self.link.resolve(strict=True)
        except OSError:
            return None

    def environments(self, with_sizes: bool = True) -> list[Environment]:
        """Every installed version, newest name last, the active one marked.

        Sizes are measured on request rather than remembered: 7.7 GB of small
        files takes a moment to add up, and this is read by a settings page
        rather than by anything on the path of a request.
        """
        here = self.active()
        try:
            entries = sorted(self.root.iterdir())
        except OSError:
            return []
        found = []
        for entry in entries:
            named = NAME.match(entry.name)
            if not entry.is_dir() or not named:
                continue
            # The original folder has no version in its name, so it is asked.
            version = "" if named.group(1) else version_in(entry, self.package)
            environment = Environment(entry, active=(here == entry.resolve()), version=version)
            if with_sizes:
                environment.size_bytes = tree_size(entry)
            found.append(environment)
        return found

    def installed_now(self) -> str:
        """The version the engine is launched from, or "" if none is."""
        if self.active() is None:
            return ""
        found = [item for item in self.environments(with_sizes=False) if item.active]
        return found[0].version if found else ""

    def point_at(self, environment: Path) -> None:
        """Make this the one the engine is launched from (`replace_link`)."""
        if not (environment / "bin").is_dir():
            raise ValueError(f"{environment.name} has no bin/ — it is not usable")
        replace_link(self.link, environment)

    def activate(self, name: str) -> None:
        """Go back — or forward — to an installed version, by folder name."""
        self.point_at(self._one(name).path)

    def remove(self, name: str) -> None:
        """Delete an installed version.

        Refused for the one in use, which would leave the engine with nothing
        to run. Nothing else is refused: keeping the previous version is the
        whole point, and when to stop keeping it is not a decision this file
        should be making.
        """
        environment = self._one(name)
        if environment.active:
            raise ValueError(
                f"{name} is the one in use. Activate another one first.")
        shutil.rmtree(environment.path)

    def _one(self, name: str) -> Environment:
        for environment in self.environments(with_sizes=False):
            if environment.name == name:
                return environment
        raise KeyError(f"No installed environment called {name}")
