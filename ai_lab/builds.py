"""Keeping an engine's source build up to date.

Both machines compile llama.cpp from git rather than installing a package, so
this module does what you would otherwise do by hand: report which build is
installed, ask upstream whether there is a newer one, and run the update while
streaming its output to the browser.

**It rebuilds; it does not reconfigure.** The existing `build/` directory
already holds the flags each machine was set up with — CUDA compiled for this
exact card on the Linux box, Metal with embedded shaders on the Mac. Running
`cmake --build` reuses them. Regenerating the configuration would mean guessing
those flags, and guessing wrong is silent: you would get a working binary that
quietly lost an optimisation.

Versions are compared using llama.cpp's own build tags, `b10331` and the like,
read straight from the remote with `git ls-remote`. That avoids the GitHub API,
which needs no key until it rate-limits you.
"""

from __future__ import annotations

import re
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from .events import EventBus
from .types import ChangeEvent, LogEvent

BUILD_TAG = re.compile(r"refs/tags/b(\d+)$")
# How often to ask upstream whether a newer build exists.
CHECK_INTERVAL_S = 3600.0
STARTUP_DELAY_S = 20.0
LOG_LINES = 500
GIT_TIMEOUT = 120
BUILD_TIMEOUT = 3600


@dataclass(frozen=True, slots=True)
class Version:
    tag: str                      # "b10331", or "" when unknown
    commit: str = ""

    @property
    def number(self) -> int:
        """The build number, for comparison. 0 when it cannot be read."""
        match = re.match(r"^b(\d+)", self.tag)
        return int(match.group(1)) if match else 0


class SourceBuild:
    """One engine's source checkout, and the ability to update it."""

    def __init__(self, engine_id: str, path: str, bus: EventBus,
                 jobs: int | None = None) -> None:
        self.engine_id = engine_id
        self.path = Path(path)
        self.bus = bus
        self.jobs = jobs
        self._lines: deque[str] = deque(maxlen=LOG_LINES)
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._state = "idle"          # idle, running, done, failed
        self._error = ""
        self._latest: Version | None = None

    # -- reading -----------------------------------------------------------

    def status(self) -> dict:
        installed = self.installed()
        latest = self._latest
        return {
            "engine": self.engine_id,
            "path": str(self.path),
            "exists": self.exists,
            "installed": installed.tag,
            "commit": installed.commit,
            "latest": latest.tag if latest else "",
            "update_available": self._update_available(installed, latest),
            "note": self._note(installed),
            "state": self._state,
            "error": self._error,
            "log": list(self._lines),
        }

    @staticmethod
    def _update_available(installed: Version, latest: Version | None) -> bool:
        """Only offer an update when both versions are actually known.

        An unreadable local version compares as zero, which would make every
        remote tag look newer and offer an update that means nothing. A
        shallow clone has no tags at all, and that is common — it is how you
        clone when you only ever intended to build once.
        """
        if latest is None or installed.number == 0:
            return False
        return latest.number > installed.number

    def _note(self, installed: Version) -> str:
        """A plain sentence about anything that needs fixing by hand."""
        if not self.exists:
            return "No git checkout at this path."
        if installed.number == 0:
            return ("This checkout has no version tags, usually because it was "
                    "cloned shallow. Run: git fetch --unshallow --tags")
        return ""

    @property
    def exists(self) -> bool:
        return (self.path / ".git").is_dir()

    def installed(self) -> Version:
        """What is checked out right now.

        Never raises. This is read whenever the settings screen is drawn, and a
        checkout that git cannot read should show as unknown rather than
        breaking the whole page.
        """
        if not self.exists:
            return Version("")
        try:
            tag = self._git("describe", "--tags", "--abbrev=0").strip()
            commit = self._git("rev-parse", "--short", "HEAD").strip()
        except ValueError:
            return Version("")
        return Version(tag, commit)

    def check(self) -> dict:
        """Ask the remote for its newest build tag.

        Network-bound, so it is only done when asked rather than on every page
        load.
        """
        if not self.exists:
            raise ValueError(f"No git checkout at {self.path}")
        output = self._git("ls-remote", "--tags", "--refs", "origin", "b*",
                           timeout=GIT_TIMEOUT)
        numbers = [int(match.group(1))
                   for line in output.splitlines()
                   if (match := BUILD_TAG.search(line.strip()))]
        if not numbers:
            raise ValueError("The remote reported no build tags")
        self._latest = Version(f"b{max(numbers)}")
        self._announce()
        return self.status()

    # -- updating ----------------------------------------------------------

    def update(self) -> dict:
        """Pull and rebuild, in the background.

        Returns immediately; progress arrives on the event stream.
        """
        with self._lock:
            if self._state == "running":
                raise ValueError("A build is already running")
            if not self.exists:
                raise ValueError(f"No git checkout at {self.path}")
            self._lines.clear()
            self._error = ""
            self._state = "running"
            self._thread = threading.Thread(target=self._run, daemon=True,
                                            name=f"build-{self.engine_id}")
            self._thread.start()
        return self.status()

    def _run(self) -> None:
        started = time.monotonic()

        def elapsed() -> int:
            return int((time.monotonic() - started) * 1000)

        try:
            self._say("status", f"Updating {self.path}", elapsed())
            self._stream(["git", "fetch", "--tags", "--prune"], elapsed, GIT_TIMEOUT)
            self._stream(["git", "pull", "--ff-only"], elapsed, GIT_TIMEOUT)

            before = self.installed()
            self._say("status", f"Now at {before.tag} ({before.commit})", elapsed())

            command = ["cmake", "--build", "build", "--config", "Release"]
            if self.jobs:
                command += ["-j", str(self.jobs)]
            self._say("status", "Compiling. This takes a while.", elapsed())
            self._stream(command, elapsed, BUILD_TIMEOUT)

            after = self.installed()
            self._state = "done"
            self._announce()
            self._latest = None          # force a fresh check before offering another
            self._say("status", f"Finished at {after.tag}", elapsed())
        except Exception as error:
            self._state = "failed"
            self._error = str(error) or error.__class__.__name__
            self._say("err", self._error, elapsed())
            self._announce()

    def _stream(self, command: list[str], elapsed, timeout: int) -> None:
        """Run a command, forwarding each line as it appears.

        Output is merged and read line by line rather than collected at the
        end, because a ten-minute compile with nothing on screen is
        indistinguishable from a hang.
        """
        self._say("status", "$ " + " ".join(command), elapsed())
        process = subprocess.Popen(
            command, cwd=self.path, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        deadline = time.monotonic() + timeout
        assert process.stdout is not None
        for line in process.stdout:
            self._say("out", line.rstrip("\n"), elapsed())
            if time.monotonic() > deadline:
                process.kill()
                raise TimeoutError(f"{command[0]} exceeded {timeout}s")
        code = process.wait()
        if code != 0:
            raise RuntimeError(f"{' '.join(command)} failed with exit code {code}")

    # -- internals ---------------------------------------------------------

    def _git(self, *arguments: str, timeout: int = 15) -> str:
        try:
            completed = subprocess.run(
                ["git", *arguments], cwd=self.path, capture_output=True,
                text=True, timeout=timeout)
        except (OSError, subprocess.TimeoutExpired) as error:
            raise ValueError(f"git {arguments[0]} failed: {error}") from None
        if completed.returncode != 0:
            raise ValueError(completed.stderr.strip() or
                             f"git {arguments[0]} failed")
        return completed.stdout

    def _announce(self) -> None:
        self.bus.publish(ChangeEvent(topic="engines"))

    def _say(self, stream: str, text: str, elapsed_ms: int) -> None:
        with self._lock:
            self._lines.append(text)
        self.bus.publish(LogEvent(source=self.engine_id, stream=stream,
                                  text=text, elapsed_ms=elapsed_ms))


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
                    engine_id, source["path"], bus, jobs=source.get("jobs"))

    def all(self) -> list[dict]:
        return [item.status() for item in self._builds.values()]

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
