"""Keeping an engine's compiled source versions up to date.

Both machines compile llama.cpp from git rather than installing a package, so
this module does what you would otherwise do by hand: report which build is
installed, ask upstream whether there is a newer one, and run the update while
streaming its output to the browser.

On a versioned installation every update is configured and compiled in a new
folder, verified, and only then selected through the stable `current` link.
The previous build remains untouched for rollback. Configuration supplies the
machine-specific CMake flags explicitly; guessing them would risk producing a
working binary that quietly lost an optimisation.

An installation without a configured build root keeps the original behaviour
for backwards compatibility: it checks out the selected tag and rebuilds the
existing `build/` directory in place.

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

import json
import re
import shutil
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
VERIFY_TIMEOUT = 60
CURRENT = "current"
META = ".ai-lab-build.json"
BUILD_NAME = re.compile(r"^build-(b\d+|v\d+\.\d+\.\d+)$")


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


class BuildEnvironment:
    """One compiled source version, shaped like a package environment."""

    def __init__(self, path: Path, version: Version, active: bool,
                 movable: bool = True) -> None:
        self.path, self.version, self.active = path, version, active
        self.name = path.name
        self.movable = movable
        self.size_bytes = 0

    def json(self) -> dict:
        return {"name": self.name, "version": self.version.tag,
                "commit": self.version.commit, "active": self.active,
                "size_bytes": self.size_bytes, "path": str(self.path),
                "movable": self.movable}


class SourceBuild:
    """One engine's source checkout, and the ability to update it."""

    def __init__(self, engine_id: str, path: str, bus: EventBus,
                 jobs: int | None = None, line: str = DEFAULT_LINE,
                 builds: str = "", legacy_build: str = "",
                 cmake_args: list[str] | None = None) -> None:
        self.engine_id = engine_id
        self.line = line if line in LINES else DEFAULT_LINE
        self.path = Path(path)
        self.bus = bus
        self.jobs = jobs
        self.build_root = Path(builds) if builds else None
        self.legacy_build = Path(legacy_build) if legacy_build else None
        self.cmake_args = list(cmake_args or [])
        self._lines: deque[str] = deque(maxlen=LOG_LINES)
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._state = "idle"          # idle, running, done, failed
        self._error = ""
        self._latest: Version | None = None

    # -- reading -----------------------------------------------------------

    def status(self, with_sizes: bool = True) -> dict:
        installed = self.installed()
        latest = self._latest
        environments = self.environments(with_sizes=with_sizes) if self.build_root else []
        return {
            "engine": self.engine_id,
            "kind": "source",
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
            "environments": [item.json() for item in environments],
            "spare_bytes": sum(item.size_bytes for item in environments
                               if not item.active),
            "free_bytes": _free(self.build_root) if self.build_root else 0,
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
        return self._is_ahead(latest.tag, installed.commit or installed.tag or "HEAD")

    def _is_ahead(self, tag: str, against: str = "HEAD") -> bool:
        """Would moving to this tag bring anything this checkout does not have?"""
        try:
            subprocess.run(["git", "merge-base", "--is-ancestor", tag, against],
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
        """What the engine is launched from, or the checkout on legacy setups.

        Never raises. This is read whenever the settings screen is drawn, and a
        checkout that git cannot read should show as unknown rather than
        breaking the whole page.
        """
        if self.build_root:
            active = self.active()
            if active is None:
                return Version("")
            marked = _marked(active)
            if marked.tag:
                return marked
            # A legacy build can be introduced to the versioned scheme before
            # its marker is written. It still corresponds to the checkout at
            # that moment; migration writes the marker before any update.
            if self.legacy_build and active == self.legacy_build.resolve():
                return self.checkout_version()
            return Version("")
        return self.checkout_version()

    def checkout_version(self) -> Version:
        """What source is checked out, independently of the active binary."""
        if not self.exists:
            return Version("")
        try:
            tag = self._git("describe", "--tags", "--abbrev=0").strip()
            commit = self._git("rev-parse", "--short", "HEAD").strip()
        except ValueError:
            return Version("")
        return Version(tag, commit)

    @property
    def link(self) -> Path | None:
        return self.build_root / CURRENT if self.build_root else None

    def active(self) -> Path | None:
        if self.link is None:
            return None
        try:
            return self.link.resolve(strict=True)
        except OSError:
            return None

    def environments(self, with_sizes: bool = True) -> list[BuildEnvironment]:
        """The active source build and every compiled rollback beside it."""
        if not self.build_root:
            return []
        active = self.active()
        candidates: list[tuple[Path, bool]] = []
        if self.legacy_build and self.legacy_build.is_dir():
            candidates.append((self.legacy_build, False))
        try:
            candidates.extend((entry, True) for entry in sorted(self.build_root.iterdir())
                              if entry.is_dir() and BUILD_NAME.match(entry.name))
        except OSError:
            pass
        found = []
        seen = set()
        for path, movable in candidates:
            try:
                resolved = path.resolve()
            except OSError:
                continue
            if resolved in seen:
                continue
            seen.add(resolved)
            version = _marked(path)
            if not version.tag and path == self.legacy_build and active == resolved:
                version = self.checkout_version()
            environment = BuildEnvironment(path, version, active == resolved, movable)
            if with_sizes:
                environment.size_bytes = _size(path)
            found.append(environment)
        return found

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
        return self.status(with_sizes=False)

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

        previous = self.installed()
        target_dir: Path | None = None
        created = False
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

            source = self.checkout_version()
            self._say("status", f"Now at {source.tag} ({source.commit})", elapsed())

            if self.build_root:
                target_dir = self.build_root / f"build-{target}"
                if target_dir.exists():
                    raise ValueError(
                        f"{target} is already compiled at {target_dir.name}. "
                        "Remove it first, or activate it instead of rebuilding it.")
                target_dir.parent.mkdir(parents=True, exist_ok=True)
                created = True
                configure = ["cmake", "-S", ".", "-B", str(target_dir),
                             "-DCMAKE_BUILD_TYPE=Release", *self.cmake_args]
                self._say("status", f"Configuring {target_dir.name}", elapsed())
                self._stream(configure, elapsed, BUILD_TIMEOUT)
                build_path = str(target_dir)
            else:
                build_path = "build"

            command = ["cmake", "--build", build_path, "--config", "Release"]
            if self.jobs:
                command += ["-j", str(self.jobs)]
            self._say("status", "Compiling. This takes a while.", elapsed())
            self._stream(command, elapsed, BUILD_TIMEOUT)

            if self.build_root and target_dir:
                binary = target_dir / "bin" / "llama-server"
                if not binary.is_file():
                    raise ValueError(f"The new build has no {binary}")
                self._say("status", "Checking that llama-server starts", elapsed())
                self._stream([str(binary), "--version"], elapsed, VERIFY_TIMEOUT)
                _mark(target_dir, source)
                old = self.active()
                self.point_at(target_dir)
                self._say("status", f"{self.engine_id} now runs from {target_dir.name}."
                          + (f" {old.name} is kept for rollback." if old else ""),
                          elapsed())

            after = self.installed()
            self._state = "done"
            self._latest = None          # force a fresh check before offering another
            self._announce()
            self._say("status", f"Finished at {after.tag}", elapsed())
        except Exception as error:
            self._state = "failed"
            self._error = str(error) or error.__class__.__name__
            self._say("err", self._error, elapsed())
            if created and target_dir and target_dir.exists() \
                    and target_dir != self.active():
                shutil.rmtree(target_dir, ignore_errors=True)
            # A failed versioned build did not become active. Put the checkout
            # back too, so update detection and change reading still compare
            # upstream with the binary that actually runs.
            if self.build_root and previous.tag:
                try:
                    self._git("checkout", "--force", previous.commit or previous.tag,
                              timeout=GIT_TIMEOUT)
                except ValueError:
                    pass
            self._announce()

    # -- choosing compiled versions ---------------------------------------

    def point_at(self, build: Path) -> None:
        if self.link is None:
            raise ValueError("This source build has no versioned build root")
        if not (build / "bin" / "llama-server").is_file():
            raise ValueError(f"{build.name} has no bin/llama-server")
        temporary = self.build_root / f".{CURRENT}-new"
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()
        temporary.symlink_to(build, target_is_directory=True)
        temporary.replace(self.link)

    def activate(self, name: str) -> dict:
        environment = self._one(name)
        reference = environment.version.commit or environment.version.tag
        if not reference:
            raise ValueError(f"{name} does not record which source it was built from")
        # Keep the source and the selected binary describing the same version.
        # This is a quick checkout, not a rebuild, and makes the next update
        # comparison start from what actually runs.
        previous = self.checkout_version()
        self._git("checkout", "--force", reference, timeout=GIT_TIMEOUT)
        try:
            self.point_at(environment.path)
        except Exception:
            if previous.tag:
                try:
                    self._git("checkout", "--force",
                              previous.commit or previous.tag, timeout=GIT_TIMEOUT)
                except ValueError:
                    pass
            raise
        self._announce()
        return self.status()

    def remove(self, name: str) -> dict:
        environment = self._one(name)
        if environment.active:
            raise ValueError(f"{name} is the build in use. Activate another one first.")
        shutil.rmtree(environment.path)
        self._announce()
        return self.status()

    def _one(self, name: str) -> BuildEnvironment:
        for environment in self.environments(with_sizes=False):
            if environment.name == name:
                return environment
        raise KeyError(f"No compiled build called {name}")

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


def _marked(path: Path) -> Version:
    try:
        raw = json.loads((path / META).read_text())
        return Version(str(raw.get("tag") or ""), str(raw.get("commit") or ""))
    except (OSError, ValueError, TypeError):
        return Version("")


def _mark(path: Path, version: Version) -> None:
    temporary = path / f"{META}.tmp"
    temporary.write_text(json.dumps({"tag": version.tag,
                                     "commit": version.commit}) + "\n")
    temporary.replace(path / META)


def _size(path: Path) -> int:
    total = 0
    try:
        for item in path.rglob("*"):
            try:
                if item.is_file() and not item.is_symlink():
                    total += item.stat().st_size
            except OSError:
                pass
    except OSError:
        pass
    return total


def _free(path: Path | None) -> int:
    try:
        return shutil.disk_usage(path).free if path else 0
    except OSError:
        return 0
