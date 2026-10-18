"""One task at a time in the background for one engine: a build or an install.

Held by `SourceBuild`, `PackageInstall` and `GitApplicationInstall`, which ran
the same small state machine three times (idle → running → done or failed, the
last lines of output, a thread, a lock). Here once.

It alone owns its lock, which guards the state, the error, the thread and the
count of changes in progress. `start` refuses while a task runs or a version
is being switched or deleted; `idle` refuses while a task runs and, for as long
as the change lasts, keeps a new task from starting — so `git checkout` never
runs in a tree `cmake` is compiling, and a folder an install is about to point
at is never deleted under it.
"""

from __future__ import annotations

import shutil
import threading
from collections import deque
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

LOG_LINES = 500

IDLE, RUNNING, DONE, FAILED = "idle", "running", "done", "failed"


class BackgroundJob:
    """The state of an engine's background task, and the guard around it."""

    def __init__(self, name: str, what: str) -> None:
        """`name` names the thread; `what` names the task in refusals ("a build")."""
        self.name = name
        self.what = what
        self._lines: deque[str] = deque(maxlen=LOG_LINES)
        self._lock = threading.RLock()
        self._state = IDLE
        self._error = ""
        self._thread: threading.Thread | None = None
        self._changing = 0

    @property
    def state(self) -> str:
        """idle, running, done or failed."""
        with self._lock:
            return self._state

    @property
    def error(self) -> str:
        """Why the last task failed, or ""."""
        with self._lock:
            return self._error

    @property
    def thread(self) -> threading.Thread | None:
        """The thread of the task running or last run, for whoever waits for it."""
        with self._lock:
            return self._thread

    def start(self, task: Callable[..., None], *args: object,
              check: Callable[[], None] | None = None) -> None:
        """Run `task(*args)` in the background; ValueError when one already runs.

        `check` runs first, under the lock, and may refuse by raising.
        """
        with self._lock:
            if self._state == RUNNING:
                raise ValueError(f"{self.what.capitalize()} is already running")
            if self._changing:
                raise ValueError(f"A version is being switched or removed; start {self.what} after")
            if check is not None:
                check()
            self._lines.clear()
            self._error = ""
            self._state = RUNNING
            self._thread = threading.Thread(target=task, args=args, daemon=True,
                                            name=self.name)
            self._thread.start()

    def succeeded(self) -> None:
        """The task finished well."""
        with self._lock:
            self._state = DONE

    def failed(self, error: Exception) -> str:
        """The task failed; returns the error as it is reported."""
        with self._lock:
            self._state = FAILED
            self._error = str(error) or error.__class__.__name__
            return self._error

    def say(self, text: str) -> None:
        """Keep one line of output for the settings page."""
        with self._lock:
            self._lines.append(text)

    def log(self) -> list[str]:
        """The last lines of output, oldest first."""
        with self._lock:
            return list(self._lines)

    @contextmanager
    def idle(self, change: str) -> Iterator[None]:
        """Make `change` (e.g. "switch versions") only while no task runs, and keep one from starting."""
        with self._lock:
            if self._state == RUNNING:
                raise ValueError(f"Cannot {change} while {self.what} is running")
            self._changing += 1
        try:
            yield
        finally:
            with self._lock:
                self._changing -= 1


def fail_and_clean(job: BackgroundJob, error: Exception, leftover: Path | None,
                   in_use: Path | None, say: Callable[[str], None]) -> None:
    """Say why a task failed, delete what it half-made, and only then mark it failed.

    `leftover` is the folder the task was making; it goes unless it is the one
    in use. Marked failed last, so nothing is switched or removed while the
    half-made folder is being deleted.
    """
    say(str(error) or error.__class__.__name__)
    if leftover is not None and leftover.exists() and leftover != in_use:
        shutil.rmtree(leftover, ignore_errors=True)
    job.failed(error)
