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

**There are two lines to follow, and which one is a choice.** Upstream tags
almost every commit to master as `b10448` — about seven a day, 7,160 of them in
the checkout when this was measured. On 17 August 2026 they also started a
second line, `v0.1.0` and up, in their own words "stable, slower release
cadence, recommended for downstream distribution and casual users", with
release notes worth reading. The `b` line they call "bleeding edge...
recommended for developers and technical users".

Which line this follows comes from `source.line` in the configuration:
"stable" for `vX.Y.Z`, "nightly" for `b<number>`. Stable is the default,
because an update nobody can read about is not a decision.

The two lines cannot be compared by their numbers — `b10448` and `v0.2.0` are
not on the same scale — so "is there something newer?" is asked of git
instead: is the tag we would move to already in this checkout's history? That
question is exact, it is the same question on both lines, and it stays right
when somebody switches from one to the other.
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

# The two lines, and how to recognise a tag on each. `pattern` is matched
# against a plain tag name; `sort` is what git is asked to order by, which
# differs because one is a plain number and the other is a version.
LINES = {
    "stable":  {"glob": "v*", "pattern": re.compile(r"^v\d+\.\d+\.\d+$")},
    "nightly": {"glob": "b*", "pattern": re.compile(r"^b\d+$")},
}
DEFAULT_LINE = "stable"
# How often to ask upstream whether a newer build exists.
CHECK_INTERVAL_S = 3600.0
STARTUP_DELAY_S = 20.0
LOG_LINES = 500
GIT_TIMEOUT = 120
BUILD_TIMEOUT = 3600


@dataclass(frozen=True, slots=True)
class Version:
    tag: str                      # "b10331" or "v0.2.0", or "" when unknown
    commit: str = ""

    @property
    def line(self) -> str:
        """Which of the two lines this tag belongs to, or "" for neither."""
        for name, shape in LINES.items():
            if shape["pattern"].match(self.tag or ""):
                return name
        return ""

    @property
    def number(self) -> tuple[int, ...]:
        """Enough of the tag to order two of them on the *same* line.

        Never used to compare across lines: `b10448` would sort above `v0.2.0`
        and mean nothing by it. Comparing the two lines is git's job — see
        `_is_ahead`.
        """
        return tuple(int(part) for part in re.findall(r"\d+", self.tag or ""))


class SourceBuild:
    """One engine's source checkout, and the ability to update it."""

    def __init__(self, engine_id: str, path: str, bus: EventBus,
                 jobs: int | None = None, line: str = DEFAULT_LINE) -> None:
        self.engine_id = engine_id
        self.line = line if line in LINES else DEFAULT_LINE
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
            "line": self.line,
            "lines": list(LINES),
            "update_available": self._update_available(installed, latest),
            "note": self._note(installed),
            "state": self._state,
            "error": self._error,
            "log": list(self._lines),
        }

    def _update_available(self, installed: Version, latest: Version | None) -> bool:
        """Is there something to move to?

        Asked of git rather than of the numbers, because the two lines are not
        on one scale: somebody on `b10448` who switches to the stable line is
        moving to `v0.2.0`, which is a smaller number and a newer thing.

        The question git answers is exact: is that tag already part of this
        checkout's history? If it is, there is nothing to do. If it is not,
        there is.

        A checkout with no tags at all — a shallow clone, which is how you
        clone when you only ever meant to build once — cannot answer, and an
        update is not offered rather than offered meaninglessly.
        """
        if latest is None or not latest.tag or not installed.tag:
            return False
        return self._is_ahead(latest.tag)

    def _is_ahead(self, tag: str) -> bool:
        """Would moving to this tag bring anything this checkout does not have?"""
        try:
            subprocess.run(["git", "merge-base", "--is-ancestor", tag, "HEAD"],
                           cwd=self.path, capture_output=True, timeout=15,
                           check=True)
        except subprocess.CalledProcessError:
            return True          # not an ancestor: there is something new
        except (OSError, subprocess.TimeoutExpired):
            return False         # cannot tell, so do not claim there is
        return False

    def _note(self, installed: Version) -> str:
        """A plain sentence about anything that needs fixing by hand."""
        if not self.exists:
            return "No git checkout at this path."
        if not installed.tag:
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
        """Fetch, then take the newest tag on the line being followed.

        The tags are fetched rather than merely listed on the remote, because
        the "is there anything newer?" question is answered by asking git
        whether that tag is already in this history — and it cannot answer
        about a tag it does not have.

        Network-bound, so it is only done when asked or on the slow timer,
        never on a page load.
        """
        if not self.exists:
            raise ValueError(f"No git checkout at {self.path}")
        self._git("fetch", "--tags", "--prune", timeout=GIT_TIMEOUT)
        newest = self.newest_on(self.line)
        if not newest:
            raise ValueError(
                f"The remote has no {self.line} tags"
                + (". Upstream only began publishing stable versions in "
                   "August 2026." if self.line == "stable" else ""))
        self._latest = Version(newest)
        self._announce()
        return self.status()

    def newest_on(self, line: str) -> str:
        """The highest tag on one line, or "" if that line has none here.

        Sorted by git rather than in Python, so that `v0.1.10` comes after
        `v0.1.2` and `b10000` after `b9999` without this file having to know
        how either is spelled.
        """
        shape = LINES.get(line) or LINES[DEFAULT_LINE]
        try:
            output = self._git("for-each-ref", "--sort=-v:refname",
                               "--format=%(refname:short)",
                               f"refs/tags/{shape['glob']}", timeout=30)
        except ValueError:
            return ""
        for candidate in output.splitlines():
            if shape["pattern"].match(candidate.strip()):
                return candidate.strip()
        return ""

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

            # Move to a named tag rather than pulling whatever master has
            # reached. On the stable line the two are not the same thing —
            # `v0.2.0` is a chosen commit some way behind the tip — and even
            # on the nightly line this makes the build reproducible: what got
            # compiled has a name, and going back means naming the old one.
            #
            # This leaves git on no branch, which is correct: the checkout is
            # at a released version rather than following a moving one.
            target = self.newest_on(self.line)
            if not target:
                raise ValueError(f"No {self.line} tag to move to")
            self._say("status", f"Moving to {target}", elapsed())
            self._stream(["git", "checkout", "--force", target],
                         elapsed, GIT_TIMEOUT)

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
                    engine_id, source["path"], bus, jobs=source.get("jobs"),
                    line=source.get("line", DEFAULT_LINE))

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
