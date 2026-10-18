"""What every music server's `main` does besides building its backend.

Each `*_server.py` is started by the manager with its settings on the command
line, all required. Some also start a companion program (the engine's own web
page, a worker) that must be ready before requests are answered, and stopped
when the server stops. Those three steps are written once here.
"""

from __future__ import annotations

import argparse
import atexit
import signal
import subprocess
import time
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# One required command-line option: its flag and how its text is read.
Option = tuple[str, Callable[[str], Any]]

PORT: Option = ("--port", int)
UI_PORT: Option = ("--ui-port", int)
# A model bundle and its checkpoint (`bundles.Checkpoint`), and token sampling
# (`bundles.Sampling`), as the manager passes them.
CHECKPOINT_OPTIONS: tuple[Option, ...] = (
    ("--bundle-root", Path), ("--checkpoint-path", Path), ("--checkpoint-subdir", str))
SAMPLING_OPTIONS: tuple[Option, ...] = (
    ("--topk", int), ("--temperature", float), ("--cfg-scale", float))

# How often a companion is asked whether it is ready.
POLL_S = 1.0
# How long one readiness question may take.
ASK_S = 2.0


def parse(*options: Option, parser: argparse.ArgumentParser | None = None) -> argparse.Namespace:
    """The command line, every option in `options` required."""
    parser = parser or argparse.ArgumentParser()
    for flag_name, kind in options:
        parser.add_argument(flag_name, type=kind, required=True)
    return parser.parse_args()


def stop_on_exit(stop: Callable[[], None]) -> None:
    """Run `stop` when the process exits, or is told to stop (SIGTERM, SIGINT)."""
    atexit.register(stop)

    def stopped(_signum, _frame) -> None:
        stop()
        raise SystemExit(0)
    signal.signal(signal.SIGTERM, stopped)
    signal.signal(signal.SIGINT, stopped)


def terminate(child: subprocess.Popen, grace_s: float) -> None:
    """Stop a companion: politely, then for good after `grace_s`."""
    if child.poll() is not None:
        return
    child.terminate()
    try:
        child.wait(timeout=grace_s)
    except subprocess.TimeoutExpired:
        child.kill()


@dataclass(frozen=True)
class Readiness:
    """How to tell a companion is ready: an address, and what its answer must say.

    `accept(response)` reads the open response and says whether it is ready;
    left out, any answer at all is enough.
    """

    url: str
    timeout_s: float
    accept: Callable[[Any], bool] | None = None


def wait_until_ready(readiness: Readiness, name: str,
                     child: subprocess.Popen | None = None) -> None:
    """Ask until the companion answers as ready; RuntimeError if it exits or never does."""
    deadline = time.monotonic() + readiness.timeout_s
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if child is not None and child.poll() is not None:
            raise RuntimeError(f"{name} exited during startup")
        try:
            with urllib.request.urlopen(readiness.url, timeout=ASK_S) as response:
                if readiness.accept is None or readiness.accept(response):
                    return
        except Exception as error:
            last_error = error
        time.sleep(POLL_S)
    raise RuntimeError(f"{name} did not become ready at {readiness.url}: {last_error}")
