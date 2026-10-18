"""What an install in the background says while it runs.

Shared by the package installs (`package_install.py`) and the Git
applications (`gitapps.py`): the last lines of output kept for the settings
page, each line also published on the event stream, and a command run with its
output forwarded line by line — a ten-minute install with nothing on screen is
indistinguishable from a hang.
"""

from __future__ import annotations

import subprocess
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
    """One program to run: its arguments, where, with which environment, for how long."""

    argv: list[str]
    cwd: Path
    environment: dict[str, str]
    timeout_s: int


class InstallLog:
    """One engine's install output: kept, and published as it arrives."""

    def __init__(self, engine_id: str, bus: EventBus | None) -> None:
        self.engine_id = engine_id
        self.bus = bus
        self.lines: deque[str] = deque(maxlen=LOG_LINES)

    def say(self, stream: str, text: str, elapsed_ms: int) -> None:
        """Keep one line and publish it; `stream` is "status", "out" or "err"."""
        self.lines.append(text)
        if self.bus:
            self.bus.publish(LogEvent(source=self.engine_id, stream=stream,
                                      text=text, elapsed_ms=elapsed_ms))

    def announce(self) -> None:
        """Tell the settings page the installs changed."""
        if self.bus:
            self.bus.publish(ChangeEvent(topic="builds"))

    def run(self, command: Command, elapsed: Elapsed) -> None:
        """Run `command`, forwarding each line; ValueError if it fails or runs too long."""
        process = subprocess.Popen(command.argv, cwd=str(command.cwd), stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True,
                                   env=command.environment)
        deadline = time.monotonic() + command.timeout_s
        assert process.stdout is not None
        for line in process.stdout:
            self.say("out", line.rstrip(), elapsed())
            if time.monotonic() > deadline:
                process.kill()
                raise ValueError(f"{command.argv[0]} took longer than {command.timeout_s}s")
        if process.wait() != 0:
            raise ValueError(f"{' '.join(command.argv[:3])} failed")


def since(started: float) -> Elapsed:
    """Milliseconds since `started` (a `time.monotonic()` reading), as a function."""
    return lambda: int((time.monotonic() - started) * 1000)
