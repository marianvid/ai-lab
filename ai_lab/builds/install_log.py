"""What an install in the background says while it runs.

Shared by the package installs (`package_install.py`) and the Git
applications (`gitapps.py`): the last lines of output kept for the settings
page, each line also published on the event stream, and a command run with its
output forwarded line by line — a ten-minute install with nothing on screen is
indistinguishable from a hang.
"""

from __future__ import annotations

import contextlib
import os
import queue
import signal
import subprocess
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ..events import EventBus
from ..types import ChangeEvent, LogEvent

LOG_LINES = 500

Elapsed = Callable[[], int]


@dataclass(frozen=True)
class Command:
    """One program to run: its arguments, where, with which environment, for how long.

    An empty `environment` means the gateway's own (the process inherits it).
    """

    argv: list[str]
    cwd: Path
    environment: dict[str, str]
    timeout_s: float


class InstallLog:
    """One engine's install output: kept, and published as it arrives."""

    def __init__(self, engine_id: str, bus: EventBus | None,
                 keep: Callable[[str], None] | None = None) -> None:
        """`keep` stores each line (a `BackgroundJob`'s `say`, under the job's lock).

        Without one the log keeps its own last lines, read with `kept`.
        """
        self.engine_id = engine_id
        self.bus = bus
        self._own: deque[str] = deque(maxlen=LOG_LINES)
        self._keep = keep or self._own.append

    def kept(self) -> list[str]:
        """The lines this log kept itself (empty when they go to a job)."""
        return list(self._own)

    def say(self, stream: str, text: str, elapsed_ms: int) -> None:
        """Keep one line and publish it; `stream` is "status", "out" or "err"."""
        self._keep(text)
        if self.bus:
            self.bus.publish(LogEvent(source=self.engine_id, stream=stream,
                                      text=text, elapsed_ms=elapsed_ms))

    def announce(self) -> None:
        """Tell the settings page the installs changed."""
        if self.bus:
            self.bus.publish(ChangeEvent(topic="builds"))

    def run(self, command: Command, elapsed: Elapsed) -> None:
        """Run `command`, forwarding each line; ValueError if it fails or runs too long."""
        try:
            code = run_streamed(command, lambda line: self.say("out", line.rstrip(), elapsed()))
        except TimeoutError:
            raise ValueError(f"{command.argv[0]} took longer than {command.timeout_s:g}s") from None
        if code != 0:
            raise ValueError(f"{' '.join(command.argv[:3])} failed")


# How long the reader of a killed command's output is waited for before its
# pipe is closed anyway (a grandchild may still hold the other end).
READER_GRACE_S = 5.0


def run_streamed(command: Command, on_line: Callable[[str], None]) -> int:
    """Run `command` to the end, each output line to `on_line`; its exit code.

    The output is read on a thread of its own, so the time limit holds while
    the command prints nothing, and holds after it closed its output too.
    The command runs in a process group of its own: a build is a tree
    (`make`, a compiler under it, `pip` building a wheel), and past
    `timeout_s` the whole tree is killed, not only its top, so nothing is left
    holding the pipe. Then it is waited for, its pipe closed, and
    `TimeoutError` raised. Shared by the installs and the source builds
    (`build_commands.SubprocessStream`).
    """
    process = subprocess.Popen(command.argv, cwd=str(command.cwd), stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, bufsize=1,
                               env=command.environment or None, start_new_session=True)
    lines: queue.Queue[str | None] = queue.Queue()
    reader = threading.Thread(target=_read_into, args=(process, lines), daemon=True,
                              name=f"output-of-{command.argv[0]}")
    reader.start()
    deadline = time.monotonic() + command.timeout_s
    try:
        while (line := _next_line(lines, deadline)) is not None:
            on_line(line.rstrip("\n"))
        try:
            return process.wait(timeout=max(deadline - time.monotonic(), 0))
        except subprocess.TimeoutExpired:
            raise TimeoutError from None
    finally:
        _end_group(process)
        process.wait()
        reader.join(READER_GRACE_S)
        if process.stdout is not None:
            process.stdout.close()


def _end_group(process: subprocess.Popen) -> None:
    """Kill every process of the command's group still alive; nothing when all have ended."""
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(process.pid, signal.SIGKILL)


def _read_into(process: subprocess.Popen, lines: queue.Queue) -> None:
    """Hand each line of the process's output on; None once it has ended."""
    try:
        if process.stdout is not None:
            for line in process.stdout:
                lines.put(line)
    except (OSError, ValueError):
        pass  # the pipe was closed under the reader: the command was ended
    finally:
        lines.put(None)


def _next_line(lines: queue.Queue, deadline: float) -> str | None:
    """The next line, None at the end; TimeoutError when the deadline comes first."""
    left = deadline - time.monotonic()
    if left <= 0:
        raise TimeoutError
    try:
        return lines.get(timeout=left)
    except queue.Empty:
        raise TimeoutError from None


def since(started: float) -> Elapsed:
    """Milliseconds since `started` (a `time.monotonic()` reading), as a function."""
    return lambda: int((time.monotonic() - started) * 1000)
