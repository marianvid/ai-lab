"""Running the long commands of a source update, one output line at a time.

A collaborator of `SourceBuild`, given through its constructor (`BuildTools`):
by default the commands run as processes; a test hands in a fake that records
them. It keeps no state, so it needs no lock.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .install_log import Command, run_streamed
from .source_checkout import GitCommands


class CommandStream(Protocol):
    """Running one command, handing on each line of its output as it appears."""

    def run(self, command: list[str], cwd: Path, on_line: Callable[[str], None],
            timeout: float) -> None:
        """Run to the end; raise if it fails or takes longer than `timeout` seconds."""
        ...


class SubprocessStream:
    """The command run as a process, output merged and read line by line.

    Read line by line rather than collected at the end, because a ten-minute
    compile with nothing on screen is indistinguishable from a hang.
    """

    def run(self, command: list[str], cwd: Path, on_line: Callable[[str], None],
            timeout: float) -> None:
        """Run to the end; raise if it fails or takes longer than `timeout` seconds.

        The time limit holds while the command is silent too (`run_streamed`).
        """
        try:
            code = run_streamed(Command(command, cwd, {}, timeout), on_line)
        except TimeoutError:
            raise TimeoutError(f"{command[0]} exceeded {timeout:g}s") from None
        if code != 0:
            raise RuntimeError(f"{' '.join(command)} failed with exit code {code}")


@dataclass(frozen=True)
class BuildTools:
    """What a source build runs: git, and the long commands. None means the real ones."""

    git: GitCommands | None = None
    commands: CommandStream | None = None
