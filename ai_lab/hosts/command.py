"""Running external commands.

One place for the timeout, the encoding and the "never raise" policy, so the
host implementations stay readable. Telemetry calls these several times a
second during a load, so a hung command must not block the caller.
"""

from __future__ import annotations

import shutil
import subprocess


class Result:
    __slots__ = ("returncode", "stdout", "stderr")

    def __init__(self, returncode: int, stdout: str, stderr: str) -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr

    @property
    def ok(self) -> bool:
        return self.returncode == 0


def run(argv: list[str], timeout: float = 10.0) -> Result:
    """Run a command and capture its output.

    A missing binary or a timeout comes back as a failed Result rather than an
    exception: every caller here treats "could not ask" and "answered no" the
    same way.
    """
    try:
        completed = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError:
        return Result(127, "", f"{argv[0]}: not found")
    except subprocess.TimeoutExpired:
        return Result(124, "", f"{argv[0]}: timed out after {timeout}s")
    except OSError as error:
        return Result(1, "", str(error))
    return Result(completed.returncode, completed.stdout, completed.stderr)


def which(name: str) -> str | None:
    return shutil.which(name)
