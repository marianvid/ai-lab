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
import urllib.error
import urllib.request
import shutil
import subprocess
import threading
import time
from collections import deque
from pathlib import Path

from .events import EventBus
from .gitapps import GitApplicationInstall
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

# Where to ask what the newest release is. One request, 251 ms measured on the
# container, against 2.2 s for a full package resolution — which is why the
# question can be asked on a timer at all, and why the page can say what is
# waiting without anybody pressing anything.
INDEX = "https://pypi.org/pypi/{package}/json"
INDEX_TIMEOUT_S = 20
CHECK_INTERVAL_S = 3600.0
STARTUP_DELAY_S = 25.0


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
                 uv: str = "uv", python: str = "", install: str = "",
                 modules: list[str] | None = None,
                 requirements: list[str] | None = None,
                 requires_cuda: bool = False,
                 pip_args: list[str] | None = None,
                 minimum_versions: dict[str, str] | None = None) -> None:
        self.engine_id = engine_id
        self.root = Path(root)
        self.package = package
        # Distribution names and import names are not the same thing. NeMo is
        # installed as `nemo_toolkit[asr]` but verified by importing
        # `nemo.collections.asr`; pyannote has the same distinction.
        self.install_spec = install or package
        self.modules = list(modules or [_module_of(package)])
        self.requirements = list(requirements or [])
        self.requires_cuda = requires_cuda
        self.pip_args = list(pip_args or [])
        self.minimum_versions = dict(minimum_versions or {})
        self.bus = bus
        self.uv = uv
        self.python = python          # which Python to build a new one from
        self._lines: deque[str] = deque(maxlen=LOG_LINES)
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._state = "idle"          # idle, running, done, failed
        self._error = ""
        self._latest = ""             # what the index last said

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

    def installed_now(self) -> str:
        """The version the engine is launched from, or "" if none is."""
        here = self.active()
        if here is None:
            return ""
        found = [item for item in self.environments(with_sizes=False)
                 if item.active]
        return found[0].version if found else ""

    def check(self) -> dict:
        """Ask the index what the newest release is.

        Network-bound, so it runs on the timer or when asked, never on a page
        load. A failure is remembered as "unknown" rather than as "up to date":
        an engine nobody could ask about must not look like one with nothing
        waiting.
        """
        try:
            with urllib.request.urlopen(
                    INDEX.format(package=self.package),
                    timeout=INDEX_TIMEOUT_S) as answer:
                self._latest = (json.loads(answer.read())
                                .get("info", {}).get("version") or "")
        except Exception:
            self._latest = ""
        self._announce()
        return self.status()

    def status(self) -> dict:
        environments = self.environments()
        installed = self.installed_now()
        return {
            "engine": self.engine_id,
            "root": str(self.root),
            "linked": bool(self.active()),
            "installed": installed,
            "latest": self._latest,
            # Only when both are known and they differ. A version that could
            # not be read is not an update waiting.
            "update_available": bool(installed and self._latest
                                     and installed != self._latest),
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
        created = False
        try:
            landed = version or self._latest or self._latest_version()
            if not landed:
                raise ValueError(f"Could not determine which {self.package} version to install")
            wanted = _pin(self.install_spec, landed)
            # A newly configured engine has no root yet. `uv venv` creates the
            # environment itself, but not an absent parent directory; failing
            # here made every first package install stop before downloading.
            self.root.mkdir(parents=True, exist_ok=True)
            target = self.root / f"{PREFIX}{landed}"
            if target.exists():
                raise ValueError(
                    f"{landed} is already installed at {target.name}. "
                    "Remove it first, or activate it instead of installing it.")

            # A virtual environment must be created at its final path: launcher
            # scripts contain that absolute path. Building elsewhere and then
            # renaming produces an environment whose shebangs point nowhere.
            building = target

            self._say("status", f"Creating an environment in {building}", elapsed())
            create = [self.uv, "venv", str(building)]
            if self.python:
                create += ["--python", self.python]
            # The target was proven absent above, so from here on any content
            # at that path belongs to this attempt, including a partial folder
            # left by `uv venv` itself failing.
            created = True
            self._stream(create, elapsed, UV_TIMEOUT_S)

            self._say("status", f"Installing {wanted}. This downloads gigabytes.",
                      elapsed())
            self._stream([self.uv, "pip", "install", wanted,
                          *self.requirements, *self.pip_args], elapsed,
                         INSTALL_TIMEOUT_S, venv=building)

            installed = _version_in(building, self.package)
            if not installed:
                raise ValueError(
                    f"{self.package} is not in the new environment after installing")
            if installed != landed:
                raise ValueError(
                    f"Asked for {landed}, but {installed} was installed")

            # Started before anything is swapped. An environment that will not
            # import is not one to point the engine at, and finding that out
            # after the swap would mean finding it out from a failed load.
            self._say("status", "Checking that it starts", elapsed())
            self._verify(building, elapsed)

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
            if created and target is not None and target.exists() and target != self.active():
                shutil.rmtree(target, ignore_errors=True)
        self._announce()

    def _latest_version(self) -> str:
        """Read the newest version when install was called before the timer."""
        try:
            with urllib.request.urlopen(
                    INDEX.format(package=self.package),
                    timeout=INDEX_TIMEOUT_S) as answer:
                return (json.loads(answer.read()).get("info", {}).get("version") or "")
        except Exception:
            return ""

    def _verify(self, environment: Path, elapsed) -> None:
        """Refuse to hand the engine something that will not start."""
        imports = ";".join(
            f"importlib.import_module({module!r})" for module in self.modules)
        versions = ";".join(
            "assert Version(meta.version(%r)) >= Version(%r), "
            "%r" % (package, minimum,
                     f"{package} must be at least {minimum}")
            for package, minimum in self.minimum_versions.items())
        cuda = ("import torch;assert torch.cuda.is_available(),"
                "'CUDA is not available'" if self.requires_cuda else "")
        program = ";".join(part for part in (
            "import importlib, importlib.metadata as meta",
            "from packaging.version import Version",
            imports,
            versions,
            cuda,
            f"print('ok', meta.version({self.package!r}))",
        ) if part)
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


def _module_of(package: str) -> str:
    return package.replace("-", "_")


def _pin(spec: str, version: str) -> str:
    """`nemo_toolkit[asr]` -> `nemo_toolkit[asr]==3.0.2`."""
    return f"{spec}=={version}"


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
