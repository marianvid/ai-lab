"""One source checkout: update, build, verify and select a version.

`SourceBuild` is the checkout as the settings page sees it. What git says
about it is `source_checkout.GitCheckout`; the compiled versions and which one
runs are `source_compiled.CompiledVersions`; both are base classes of it.
This file is the update itself: fetch, move to the newest tag on the line
followed, compile (into its own folder when builds are versioned), check that
the new binary starts, and switch to it — or put everything back on failure.
"""

from __future__ import annotations

import shutil
import subprocess
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from ..disk import free_bytes
from ..events import EventBus
from .source_checkout import GitCheckout
from .source_compiled import CHECKOUT_S, CompiledVersions
from .source_versions import DEFAULT_LINE, LINES, Version, mark_version, marked_version
from ..types import ChangeEvent, LogEvent

LOG_LINES = 500
BUILD_TIMEOUT = 3600
VERIFY_TIMEOUT = 60

Elapsed = Callable[[], int]


@dataclass(frozen=True)
class BuildOptions:
    """How one checkout is built: from the engine's `source` settings.

    `builds` is the versioned build root ("" for a single in-tree `build/`),
    `legacy_build` an in-tree build adopted into the versioned scheme.
    """

    jobs: int | None = None
    line: str = DEFAULT_LINE
    builds: str = ""
    legacy_build: str = ""
    cmake_args: list[str] = field(default_factory=list)

    @classmethod
    def from_source(cls, source: dict) -> BuildOptions:
        """Read from an engine's `source` section."""
        return cls(jobs=source.get("jobs"), line=source.get("line", DEFAULT_LINE),
                   builds=source.get("builds", ""),
                   legacy_build=source.get("legacy_build", ""),
                   cmake_args=list(source.get("cmake_args") or []))


@dataclass
class _Attempt:
    """One update in progress: when it began, where it started, what it made."""

    started: float
    previous: Version
    target_dir: Path | None = None
    created: bool = False

    def elapsed(self) -> int:
        """Milliseconds since the update began."""
        return int((time.monotonic() - self.started) * 1000)


class SourceBuild(GitCheckout, CompiledVersions):
    """One engine's source checkout, and the ability to update it."""

    def __init__(self, engine_id: str, path: str, bus: EventBus,
                 options: BuildOptions | None = None) -> None:
        options = options or BuildOptions()
        self.engine_id = engine_id
        self.line = options.line if options.line in LINES else DEFAULT_LINE
        self.path = Path(path)
        self.bus = bus
        self.jobs = options.jobs
        self.build_root = Path(options.builds) if options.builds else None
        self.legacy_build = Path(options.legacy_build) if options.legacy_build else None
        self.cmake_args = list(options.cmake_args)
        self._lines: deque[str] = deque(maxlen=LOG_LINES)
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._state = "idle"          # idle, running, done, failed
        self._error = ""
        self._latest: Version | None = None

    # -- reading -----------------------------------------------------------

    def status(self, with_sizes: bool = True) -> dict:
        """Everything the settings page shows for this checkout."""
        installed = self.installed()
        latest = self._latest
        environments = self.environments(with_sizes=with_sizes) if self.build_root else []
        return {
            "engine": self.engine_id, "kind": "source", "path": str(self.path),
            "exists": self.exists, "installed": installed.tag, "commit": installed.commit,
            "latest": latest.tag if latest else "", "line": self.line,
            "lines": list(LINES),
            "update_available": self._update_available(installed, latest),
            "note": self._note(installed), "state": self._state, "error": self._error,
            "log": list(self._lines),
            "environments": [item.json() for item in environments],
            "spare_bytes": sum(item.size_bytes for item in environments
                               if not item.active),
            "free_bytes": free_bytes(self.build_root),
        }

    def installed(self) -> Version:
        """What the engine is launched from, or the checkout on legacy setups.

        Never raises. This is read whenever the settings screen is drawn, and a
        checkout that git cannot read should show as unknown rather than
        breaking the whole page.
        """
        if not self.build_root:
            return self.checkout_version()
        active = self.active()
        if active is None:
            return Version("")
        marked = marked_version(active)
        if marked.tag:
            return marked
        # A legacy build can be introduced to the versioned scheme before
        # its marker is written. It still corresponds to the checkout at
        # that moment; migration writes the marker before any update.
        if self.legacy_build and active == self.legacy_build.resolve():
            return self.checkout_version()
        return Version("")

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
        self._git("fetch", "--tags", "--prune", timeout=CHECKOUT_S)
        newest = self.newest_on(self.line)
        if not newest:
            raise ValueError(
                f"The remote has no {self.line} tags"
                + (". Upstream only began publishing stable versions in "
                   "August 2026." if self.line == "stable" else ""))
        self._latest = Version(newest)
        self._announce()
        return self.status(with_sizes=False)

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
        attempt = _Attempt(time.monotonic(), self.installed())
        try:
            target, source = self._move_to_newest(attempt)
            self._compile(target, attempt)
            if attempt.target_dir:
                self._verify_and_select(attempt.target_dir, source, attempt)
            after = self.installed()
            self._state = "done"
            self._latest = None          # force a fresh check before offering another
            self._announce()
            self._say("status", f"Finished at {after.tag}", attempt.elapsed())
        except Exception as error:
            self._failed(error, attempt)

    def _move_to_newest(self, attempt: _Attempt) -> tuple[str, Version]:
        """Fetch, and check out the newest tag on the line followed.

        A named tag rather than whatever master has reached. On the stable
        line the two are not the same thing — `v0.2.0` is a chosen commit some
        way behind the tip — and even on the nightly line this makes the build
        reproducible: what got compiled has a name, and going back means
        naming the old one. Git is left on no branch, which is correct.
        """
        self._say("status", f"Updating {self.path}", attempt.elapsed())
        self._stream(["git", "fetch", "--tags", "--prune"], attempt.elapsed, CHECKOUT_S)
        target = self.newest_on(self.line)
        if not target:
            raise ValueError(f"No {self.line} tag to move to")
        self._say("status", f"Moving to {target}", attempt.elapsed())
        self._stream(["git", "checkout", "--force", target], attempt.elapsed, CHECKOUT_S)
        source = self.checkout_version()
        self._say("status", f"Now at {source.tag} ({source.commit})", attempt.elapsed())
        return target, source

    def _compile(self, target: str, attempt: _Attempt) -> None:
        """Configure a new versioned build folder (if builds are versioned), then compile."""
        build_path = "build"
        if self.build_root:
            target_dir = self.build_root / f"build-{target}"
            if target_dir.exists():
                raise ValueError(
                    f"{target} is already compiled at {target_dir.name}. "
                    "Remove it first, or activate it instead of rebuilding it.")
            target_dir.parent.mkdir(parents=True, exist_ok=True)
            attempt.target_dir, attempt.created = target_dir, True
            self._say("status", f"Configuring {target_dir.name}", attempt.elapsed())
            self._stream(["cmake", "-S", ".", "-B", str(target_dir),
                          "-DCMAKE_BUILD_TYPE=Release", *self.cmake_args],
                         attempt.elapsed, BUILD_TIMEOUT)
            build_path = str(target_dir)
        command = ["cmake", "--build", build_path, "--config", "Release"]
        if self.jobs:
            command += ["-j", str(self.jobs)]
        self._say("status", "Compiling. This takes a while.", attempt.elapsed())
        self._stream(command, attempt.elapsed, BUILD_TIMEOUT)

    def _verify_and_select(self, target_dir: Path, source: Version, attempt: _Attempt) -> None:
        """Check that the new llama-server starts, record its version, and switch to it."""
        binary = target_dir / "bin" / "llama-server"
        if not binary.is_file():
            raise ValueError(f"The new build has no {binary}")
        self._say("status", "Checking that llama-server starts", attempt.elapsed())
        self._stream([str(binary), "--version"], attempt.elapsed, VERIFY_TIMEOUT)
        mark_version(target_dir, source)
        old = self.active()
        self.point_at(target_dir)
        self._say("status", f"{self.engine_id} now runs from {target_dir.name}."
                  + (f" {old.name} is kept for rollback." if old else ""), attempt.elapsed())

    def _failed(self, error: Exception, attempt: _Attempt) -> None:
        """Say why, delete the half-made build, and put the checkout back."""
        self._state = "failed"
        self._error = str(error) or error.__class__.__name__
        self._say("err", self._error, attempt.elapsed())
        target_dir = attempt.target_dir
        if attempt.created and target_dir and target_dir.exists() \
                and target_dir != self.active():
            shutil.rmtree(target_dir, ignore_errors=True)
        # A failed versioned build did not become active. Put the checkout
        # back too, so update detection and change reading still compare
        # upstream with the binary that actually runs.
        if self.build_root:
            self._back_to(attempt.previous)
        self._announce()

    def _stream(self, command: list[str], elapsed: Elapsed, timeout: int) -> None:
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

    def _announce(self) -> None:
        self.bus.publish(ChangeEvent(topic="engines"))

    def _say(self, stream: str, text: str, elapsed_ms: int) -> None:
        with self._lock:
            self._lines.append(text)
        self.bus.publish(LogEvent(source=self.engine_id, stream=stream,
                                  text=text, elapsed_ms=elapsed_ms))
