"""Installing an engine that arrives as packages, beside the one that works.

vLLM is not compiled here. It is 382 packages and 7.7 GB in a Python virtual
environment — a folder holding its own copy of Python and everything that
version needs, so two of them can sit side by side without touching each other.

**Nothing is ever installed over what works.** A new version is built in a new
folder, checked that it actually starts, and only then does the engine start
pointing at it. The old folder stays exactly as it was, so going back is
instant and is the same act as going forward.

That is not a nicety here. The vLLM installed on this machine when this was
written cannot be reinstalled: its wheel is no longer in the local cache, and
the index it came from is recorded nowhere — a nightly build with a git hash in
its name, and those indexes are pruned. An update in place would have been
irreversible.

How the engine finds the right one: it is launched through a fixed path, a link
called `current` that points at whichever folder is in use. Swapping versions
means repointing that link, which happens in one step — there is no moment when
it points at nothing.

Two folders is the steady state, about 16 GB: the one in use and the one before
it. It does not grow on its own, and it is never tidied automatically — the old
one *is* the way back, so it goes when somebody decides it can, not when a rule
decides for them.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import threading
import time
from collections import deque
from pathlib import Path

from .events import EventBus
from .types import ChangeEvent, LogEvent

# The link the engine is launched through. Everything else is a sibling of it.
CURRENT = "current"

# What a version folder is called. The version is in the name so that the list
# reads as versions rather than as dates, and `.venv-` marks them as ours —
# nothing else in that directory is touched.
#
# Plain `.venv` is accepted as well, for the environment that was there before
# any of this existed. It cannot be renamed into the scheme: a virtual
# environment's launcher scripts carry the absolute path they were built at in
# their first line, so `/opt/ai/vllm/.venv/bin/vllm` starts
# `/opt/ai/vllm/.venv/bin/python` by name and moving the folder breaks it. New
# ones are created at their final path and have no such problem. Its version is
# read from the environment itself instead of from its name.
PREFIX = ".venv-"
NAME = re.compile(r"^\.venv(?:-(.+))?$")
FIRST = ".venv"

LOG_LINES = 500
INSTALL_TIMEOUT_S = 3600      # 8 GB of wheels over a home line
CHECK_TIMEOUT_S = 180         # importing torch and vllm is not quick
UV_TIMEOUT_S = 60


class Environment:
    """One installed version, as a folder on disk."""

    __slots__ = ("path", "name", "version", "active", "size_bytes", "movable")

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
        return {"name": self.name, "version": self.version,
                "active": self.active, "size_bytes": self.size_bytes,
                "path": str(self.path), "movable": self.movable}


class PackageInstall:
    """The installed versions of one engine, and the ability to add or drop one.

    `root` is the directory holding them. `package` is what to install — for
    vLLM, "vllm". `uv` is the program that does the installing; it is asked for
    by name so a machine that keeps it somewhere unusual can say where.
    """

    def __init__(self, engine_id: str, root: str, package: str, bus: EventBus,
                 uv: str = "uv", python: str = "") -> None:
        self.engine_id = engine_id
        self.root = Path(root)
        self.package = package
        self.bus = bus
        self.uv = uv
        self.python = python          # which Python to build a new one from
        self._lines: deque[str] = deque(maxlen=LOG_LINES)
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._state = "idle"          # idle, running, done, failed
        self._error = ""

    # -- what is here ------------------------------------------------------

    @property
    def link(self) -> Path:
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
        found = []
        try:
            entries = sorted(self.root.iterdir())
        except OSError:
            return []
        for entry in entries:
            if not entry.is_dir() or not NAME.match(entry.name):
                continue
            # The original folder has no version in its name, so it is asked.
            named = NAME.match(entry.name)
            version = ("" if named and named.group(1)
                       else _version_in(entry, self.package))
            environment = Environment(entry, active=(here == entry.resolve()),
                                      version=version)
            if with_sizes:
                environment.size_bytes = _size(entry)
            found.append(environment)
        return found

    def status(self) -> dict:
        environments = self.environments()
        return {
            "engine": self.engine_id,
            "root": str(self.root),
            "linked": bool(self.active()),
            "environments": [item.json() for item in environments],
            "spare_bytes": sum(item.size_bytes for item in environments
                               if not item.active),
            "free_bytes": _free(self.root),
            "state": self._state,
            "error": self._error,
            "log": list(self._lines),
        }

    # -- adding one --------------------------------------------------------

    def install(self, version: str = "") -> dict:
        """Build a new environment beside the current one, in the background.

        `version` empty means whatever the package manager resolves to newest.
        Returns immediately; progress arrives on the event stream.
        """
        with self._lock:
            if self._state == "running":
                raise ValueError("An install is already running")
            self._lines.clear()
            self._error = ""
            self._state = "running"
            self._thread = threading.Thread(
                target=self._run, args=(version,), daemon=True,
                name=f"install-{self.engine_id}")
            self._thread.start()
        return self.status()

    def _run(self, version: str) -> None:
        started = time.monotonic()

        def elapsed() -> int:
            return int((time.monotonic() - started) * 1000)

        target: Path | None = None
        try:
            wanted = f"{self.package}=={version}" if version else self.package
            # Built somewhere temporary first, because the folder's name is the
            # version and the version is only known once it is resolved.
            building = self.root / f"{PREFIX}installing"
            if building.exists():
                shutil.rmtree(building, ignore_errors=True)

            self._say("status", f"Creating an environment in {building}", elapsed())
            create = [self.uv, "venv", str(building)]
            if self.python:
                create += ["--python", self.python]
            self._stream(create, elapsed, UV_TIMEOUT_S)

            self._say("status", f"Installing {wanted}. This downloads gigabytes.",
                      elapsed())
            self._stream([self.uv, "pip", "install", wanted], elapsed,
                         INSTALL_TIMEOUT_S, venv=building)

            landed = _version_in(building, self.package)
            if not landed:
                raise ValueError(
                    f"{self.package} is not in the new environment after installing")
            target = self.root / f"{PREFIX}{landed}"
            if target.exists():
                raise ValueError(
                    f"{landed} is already installed at {target.name}. "
                    "Remove it first, or activate it instead of installing it.")

            # Started before anything is swapped. An environment that will not
            # import is not one to point the engine at, and finding that out
            # after the swap would mean finding it out from a failed load.
            self._say("status", "Checking that it starts", elapsed())
            self._verify(building, elapsed)

            building.rename(target)
            self._say("status", f"Installed {landed} at {target.name}", elapsed())

            previous = self.active()
            self.point_at(target)
            self._state = "done"
            self._say("status", f"{self.engine_id} now runs from {target.name}."
                      + (f" {previous.name} is kept, so going back is one press."
                         if previous else ""), elapsed())
        except Exception as error:
            self._state = "failed"
            self._error = str(error) or error.__class__.__name__
            self._say("err", self._error, elapsed())
            # Whatever was half-built goes. What was working is untouched: it
            # was never written to.
            leftover = self.root / f"{PREFIX}installing"
            if leftover.exists():
                shutil.rmtree(leftover, ignore_errors=True)
        self._announce()

    def _verify(self, environment: Path, elapsed) -> None:
        """Refuse to hand the engine something that will not start."""
        program = (f"import {self.package} as engine, importlib.metadata as meta;"
                   f"print('ok', meta.version({self.package!r}))")
        result = subprocess.run([str(environment / "bin" / "python"), "-c", program],
                                capture_output=True, text=True,
                                timeout=CHECK_TIMEOUT_S)
        for line in (result.stdout + result.stderr).splitlines():
            if line.strip():
                self._say("out", line.rstrip(), elapsed())
        if result.returncode != 0:
            raise ValueError(
                f"The new environment does not start: "
                f"{(result.stderr or result.stdout).strip().splitlines()[-1]}")

    # -- choosing between them ---------------------------------------------

    def point_at(self, environment: Path) -> None:
        """Make this the one the engine is launched from.

        Done by writing a new link beside the old one and renaming it over the
        top, which the filesystem does in one step. A link that is deleted and
        then recreated has a moment in between where it points at nothing, and
        anything starting in that moment fails for a reason nobody would guess.
        """
        if not (environment / "bin").is_dir():
            raise ValueError(f"{environment.name} has no bin/ — it is not usable")
        temporary = self.root / f".{CURRENT}-new"
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()
        temporary.symlink_to(environment, target_is_directory=True)
        temporary.replace(self.link)

    def activate(self, name: str) -> dict:
        """Go back — or forward — to an installed version, by folder name."""
        environment = self._one(name)
        self.point_at(environment.path)
        self._announce()
        return self.status()

    def remove(self, name: str) -> dict:
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
        self._announce()
        return self.status()

    def _one(self, name: str) -> Environment:
        for environment in self.environments(with_sizes=False):
            if environment.name == name:
                return environment
        raise KeyError(f"No installed environment called {name}")

    # -- running things ----------------------------------------------------

    def _stream(self, command: list[str], elapsed, timeout: int,
                venv: Path | None = None) -> None:
        environment = {"PATH": "/usr/local/bin:/usr/bin:/bin",
                       "HOME": str(Path.home())}
        if venv is not None:
            environment["VIRTUAL_ENV"] = str(venv)
        process = subprocess.Popen(command, cwd=str(self.root),
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True,
                                   env=environment)
        deadline = time.monotonic() + timeout
        assert process.stdout is not None
        for line in process.stdout:
            self._say("out", line.rstrip(), elapsed())
            if time.monotonic() > deadline:
                process.kill()
                raise ValueError(f"{command[0]} took longer than {timeout}s")
        if process.wait() != 0:
            raise ValueError(f"{' '.join(command[:3])} failed")

    def _announce(self) -> None:
        if self.bus:
            self.bus.publish(ChangeEvent(topic="builds"))

    def _say(self, stream: str, text: str, elapsed_ms: int) -> None:
        self._lines.append(text)
        if self.bus:
            self.bus.publish(LogEvent(source=self.engine_id, stream=stream,
                                      text=text, elapsed_ms=elapsed_ms))


class Installs:
    """The package-installed engines this machine knows about, keyed by engine."""

    def __init__(self, settings: dict, bus: EventBus) -> None:
        self._installs: dict[str, PackageInstall] = {}
        for engine_id, engine_settings in (settings or {}).items():
            source = (engine_settings or {}).get("source") or {}
            package = source.get("package")
            root = source.get("root") or _root_of(engine_settings.get("binary", ""))
            if package and root:
                self._installs[engine_id] = PackageInstall(
                    engine_id, root, package, bus,
                    uv=source.get("uv", "uv"), python=source.get("python", ""))

    def all(self) -> list[dict]:
        return [item.status() for item in self._installs.values()]

    def get(self, engine_id: str) -> PackageInstall:
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


def _version_in(environment: Path, package: str) -> str:
    program = f"import importlib.metadata as m; print(m.version({package!r}))"
    try:
        result = subprocess.run([str(environment / "bin" / "python"), "-c", program],
                                capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def _size(path: Path) -> int:
    """How much room a folder takes, tolerating it vanishing while counted.

    A half-built environment is being deleted at the same moment the settings
    page asks how big everything is, and walking a tree that is disappearing
    raises rather than returning what it managed to see. An approximate size
    for a folder that is on its way out is the right answer; an exception that
    empties the page is not.
    """
    total = 0
    try:
        for item in path.rglob("*"):
            try:
                if item.is_file() and not item.is_symlink():
                    total += item.stat().st_size
            except OSError:
                continue
    except OSError:
        pass
    return total


def _free(path: Path) -> int:
    try:
        return shutil.disk_usage(path).free
    except OSError:
        return 0
