"""Install, verify and activate package environments beside a working one.

`PackageInstall` builds a new version in its own folder beside the one that
works, checks that it starts, and only then points the engine at it — so what
is working is never written to, and going back is always possible. It holds
which folders exist and which one runs (`package_environments.EnvironmentFolders`),
and runs its commands and its start check through `InstallTools` — real by
default, fakes in a test. It alone owns its lock.
"""

from __future__ import annotations

import json
import subprocess
import time
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from ..disk import free_bytes
from ..events import EventBus
from .background_job import BackgroundJob, fail_and_clean
from .install_log import Command, Elapsed, InstallLog, since
from .package_environments import CURRENT, PREFIX, Environment, EnvironmentFolders, version_in
from .public_services import address

__all__ = ["CURRENT", "PackageInstall", "PackageOptions"]

INSTALL_TIMEOUT_S = 3600
CHECK_TIMEOUT_S = 180
UV_TIMEOUT_S = 60
INDEX_TIMEOUT_S = 20
# The programs a new environment is built with are found on this search path.
SYSTEM_PATH = "/usr/local/bin:/usr/bin:/bin"


class InstallCommands(Protocol):
    """Running one install command (`uv venv`, `uv pip install`), its output kept."""

    def run(self, command: Command, elapsed: Elapsed) -> None:
        """Run to the end; ValueError if it fails or runs too long."""
        ...


class CheckResult(Protocol):
    """What running the start check said."""

    returncode: int
    stdout: str
    stderr: str


class EnvironmentCheck(Protocol):
    """Running a Python program with a new environment's own Python."""

    def check(self, environment: Path, program: str) -> CheckResult:
        """Run `program` there and say what happened."""
        ...


class SubprocessCheck:
    """The start check run as a process."""

    def check(self, environment: Path, program: str) -> CheckResult:
        """Run `program` with the environment's own Python."""
        return _run_check(environment, program)


@dataclass(frozen=True)
class InstallTools:
    """What an install runs: its commands and its start check. None means the real ones."""

    commands: InstallCommands | None = None
    check: EnvironmentCheck | None = None


@dataclass(frozen=True)
class PackageOptions:
    """How one engine's packages are installed and checked, from its `source` settings.

    Distribution names and import names are not the same thing: NeMo is
    installed as `nemo_toolkit[asr]` (`install`) but verified by importing
    `nemo.collections.asr` (`modules`); pyannote has the same distinction.
    `uv` is the program that installs, `python` which Python a new
    environment is built from ("" for uv's choice).
    """

    uv: str = "uv"
    python: str = ""
    install: str = ""
    modules: list[str] = field(default_factory=list)
    requirements: list[str] = field(default_factory=list)
    requires_cuda: bool = False
    pip_args: list[str] = field(default_factory=list)
    minimum_versions: dict[str, str] = field(default_factory=dict)
    # Not configuration: the collaborators that run the commands and the check.
    tools: InstallTools = field(default_factory=InstallTools)

    @classmethod
    def from_source(cls, source: dict) -> PackageOptions:
        """Read from an engine's `source` section."""
        return cls(uv=source.get("uv", "uv"), python=source.get("python", ""),
                   install=source.get("install", ""),
                   modules=list(source.get("modules") or []),
                   requirements=list(source.get("requirements") or []),
                   requires_cuda=bool(source.get("requires_cuda", False)),
                   pip_args=list(source.get("pip_args") or []),
                   minimum_versions=dict(source.get("minimum_versions") or {}))


class PackageInstall:
    """The installed versions of one engine, and the ability to add or drop one.

    `root` is the directory holding them. `package` is what to install — for
    vLLM, "vllm". The install runs as a `BackgroundJob`, which owns the only
    lock (state, error, log lines) and keeps a version from being switched or
    removed while an install runs.
    """

    def __init__(self, engine_id: str, root: str, package: str, bus: EventBus,
                 options: PackageOptions | None = None) -> None:
        options = options or PackageOptions()
        self.engine_id = engine_id
        self.root = Path(root)
        self.package = package
        self.install_spec = options.install or package
        self.modules = list(options.modules or [package.replace("-", "_")])
        self.requirements = list(options.requirements)
        self.requires_cuda = options.requires_cuda
        self.pip_args = list(options.pip_args)
        self.minimum_versions = dict(options.minimum_versions)
        self.bus = bus
        self.uv = options.uv
        self.python = options.python
        self.job = BackgroundJob(f"install-{engine_id}", "an install")
        self.log = InstallLog(engine_id, bus, self.job.say)
        self.folders = EnvironmentFolders(self.root, package)
        self.commands: InstallCommands = options.tools.commands or self.log
        self.checker: EnvironmentCheck = options.tools.check or SubprocessCheck()
        self._latest = ""             # what the index last said

    # -- the environment folders, as before -----------------------------

    @property
    def link(self) -> Path:
        """The `current` link naming the environment in use."""
        return self.folders.link

    def active(self) -> Path | None:
        """The folder the engine is currently launched from."""
        return self.folders.active()

    def installed_now(self) -> str:
        """The version the engine is launched from, or "" if none is."""
        return self.folders.installed_now()

    def environments(self, with_sizes: bool = True) -> list[Environment]:
        """Every installed version, the active one marked."""
        return self.folders.environments(with_sizes)

    def point_at(self, environment: Path) -> None:
        """Make this the one the engine is launched from."""
        self.folders.point_at(environment)

    def activate(self, name: str) -> dict:
        """Go back — or forward — to an installed version, by folder name."""
        return self._after(self.folders.activate, name, "switch versions")

    def remove(self, name: str) -> dict:
        """Delete an installed version that is not the one in use."""
        return self._after(self.folders.remove, name, "remove a version")

    def _after(self, change: Callable[[str], None], name: str, what: str) -> dict:
        """Make one change to the folders while no install runs, tell the page, say how things stand."""
        with self.job.idle(what):
            change(name)
        self._announce()
        return self.status()

    def check(self) -> dict:
        """Ask the index what the newest release is.

        Network-bound, so it runs on the timer or when asked, never on a page
        load. A failure is remembered as "unknown" rather than as "up to date":
        an engine nobody could ask about must not look like one with nothing
        waiting.
        """
        self._latest = self._latest_version()
        self._announce()
        return self.status()

    def status(self) -> dict:
        """Everything the settings page shows for this engine's installs."""
        environments = self.environments()
        installed = self.installed_now()
        return {
            "engine": self.engine_id, "root": str(self.root),
            "linked": bool(self.active()), "installed": installed,
            "latest": self._latest,
            # Only when both are known and they differ. A version that could
            # not be read is not an update waiting.
            "update_available": bool(installed and self._latest
                                     and installed != self._latest),
            "environments": [item.json() for item in environments],
            "spare_bytes": sum(item.size_bytes for item in environments
                               if not item.active),
            "free_bytes": free_bytes(self.root), "state": self.job.state,
            "error": self.job.error, "log": self.job.log(),
        }

    # -- adding one --------------------------------------------------------

    def install(self, version: str = "") -> dict:
        """Build a new environment beside the current one, in the background.

        `version` empty means whatever the package manager resolves to newest.
        Returns immediately; progress arrives on the event stream.
        """
        self.job.start(self._run, version)
        return self.status()

    def _run(self, version: str) -> None:
        elapsed = since(time.monotonic())
        target: Path | None = None
        try:
            landed = version or self._latest or self._latest_version()
            if not landed:
                raise ValueError(f"Could not determine which {self.package} version to install")
            target = self._new_folder(landed)
            # The target was proven absent, so from here on any content at that
            # path belongs to this attempt, including a partial folder left by
            # `uv venv` itself failing.
            self._build(target, landed, elapsed)
            previous = self.active()
            self.point_at(target)
            self.job.succeeded()
            self._say("status", f"{self.engine_id} now runs from {target.name}."
                      + (f" {previous.name} is kept, so going back is one press."
                         if previous else ""), elapsed())
        except Exception as error:
            # Whatever was half-built goes. What was working is untouched: it
            # was never written to.
            fail_and_clean(self.job, error, target, self.active(),
                           lambda text: self._say("err", text, elapsed()))
        self._announce()

    def _new_folder(self, landed: str) -> Path:
        """Where `landed` will be installed; refused when it is there already.

        A newly configured engine has no root yet. `uv venv` creates the
        environment itself, but not an absent parent directory.
        """
        self.root.mkdir(parents=True, exist_ok=True)
        target = self.root / f"{PREFIX}{landed}"
        if target.exists():
            raise ValueError(
                f"{landed} is already installed at {target.name}. "
                "Remove it first, or activate it instead of installing it.")
        return target

    def _build(self, target: Path, landed: str, elapsed: Elapsed) -> None:
        """Create the environment at its final path, install, and check that it starts.

        At its final path because launcher scripts contain that absolute path:
        building elsewhere and renaming produces an environment whose shebangs
        point nowhere. Checked before anything is swapped: an environment that
        will not import is not one to point the engine at.
        """
        self._say("status", f"Creating an environment in {target}", elapsed())
        create = [self.uv, "venv", str(target)]
        if self.python:
            create += ["--python", self.python]
        self._stream(create, elapsed, UV_TIMEOUT_S)
        wanted = f"{self.install_spec}=={landed}"
        self._say("status", f"Installing {wanted}. This downloads gigabytes.", elapsed())
        self._stream([self.uv, "pip", "install", wanted, *self.requirements, *self.pip_args],
                     elapsed, INSTALL_TIMEOUT_S, venv=target)
        installed = version_in(target, self.package)
        if not installed:
            raise ValueError(f"{self.package} is not in the new environment after installing")
        if installed != landed:
            raise ValueError(f"Asked for {landed}, but {installed} was installed")
        self._say("status", "Checking that it starts", elapsed())
        self._verify(target, elapsed)
        self._say("status", f"Installed {landed} at {target.name}", elapsed())

    def _latest_version(self) -> str:
        """The newest release the index lists; "" when it cannot be asked."""
        try:
            with urllib.request.urlopen(
                    address("pypi_package").format(package=self.package),
                    timeout=INDEX_TIMEOUT_S) as answer:
                return (json.loads(answer.read()).get("info", {}).get("version") or "")
        except Exception:
            return ""

    def _verify(self, environment: Path, elapsed: Elapsed) -> None:
        """Refuse to hand the engine something that will not start."""
        result = self.checker.check(environment, self._check_program())
        for line in (result.stdout + result.stderr).splitlines():
            if line.strip():
                self._say("out", line.rstrip(), elapsed())
        if result.returncode != 0:
            raise ValueError(
                f"The new environment does not start: "
                f"{(result.stderr or result.stdout).strip().splitlines()[-1]}")

    def _check_program(self) -> str:
        """A Python one-liner importing every module and checking every minimum version."""
        imports = ";".join(
            f"importlib.import_module({module!r})" for module in self.modules)
        versions = ";".join(_minimum_check(package, minimum)
                            for package, minimum in self.minimum_versions.items())
        cuda = ("import torch;assert torch.cuda.is_available(),"
                "'CUDA is not available'" if self.requires_cuda else "")
        return ";".join(part for part in (
            "import importlib, importlib.metadata as meta",
            "from packaging.version import Version",
            imports, versions, cuda,
            f"print('ok', meta.version({self.package!r}))",
        ) if part)

    # -- running things ----------------------------------------------------

    def _stream(self, command: list[str], elapsed: Elapsed, timeout: int,
                venv: Path | None = None) -> None:
        environment = {"PATH": SYSTEM_PATH, "HOME": str(Path.home())}
        if venv is not None:
            environment["VIRTUAL_ENV"] = str(venv)
        self.commands.run(Command(command, self.root, environment, timeout), elapsed)

    def _announce(self) -> None:
        self.log.announce()

    def _say(self, stream: str, text: str, elapsed_ms: int) -> None:
        self.log.say(stream, text, elapsed_ms)


def _minimum_check(package: str, minimum: str) -> str:
    """One assertion of the check program: `package` is at least `minimum`."""
    message = f"{package} must be at least {minimum}"
    return (f"assert Version(meta.version({package!r})) >= Version({minimum!r}), "
            f"{message!r}")


def _run_check(environment: Path, program: str) -> subprocess.CompletedProcess[str]:
    """Run `program` with the environment's own Python."""
    return subprocess.run([str(environment / "bin" / "python"), "-c", program],
                          capture_output=True, text=True, timeout=CHECK_TIMEOUT_S)
