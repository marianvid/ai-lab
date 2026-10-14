"""One llama.cpp server on the processor: started when wanted, stopped when idle.

The manager runs it itself, as a child process, rather than through systemd
like the card's engines: it takes no card memory, starts in a second or two,
and nothing is lost when it goes away with the manager.

`CUDA_VISIBLE_DEVICES` is emptied and no layer is offloaded (`-ngl 0`), so
the server cannot touch the card even though the binary was built with CUDA.
"""

from __future__ import annotations

import os
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ..network import LOOPBACK
from .settings import CpuModel

STARTUP_LIMIT_S = 120.0
STOP_LIMIT_S = 30.0
POLL_S = 0.25

Launcher = Callable[[list[str], dict, Path], subprocess.Popen]
Probe = Callable[[int], bool]


def command(binary: str, model: CpuModel) -> list[str]:
    """`llama-server` serving embeddings for this model, on the processor only."""
    return [binary, "-m", model.path, "--embedding", "--pooling", model.pooling,
            "-ngl", "0", "--host", LOOPBACK, "--port", str(model.port),
            "-c", str(model.context), "-b", str(model.context), "-ub", str(model.context),
            "-t", str(model.threads), "--parallel", str(model.parallel), "--no-webui"]


def _launch(argv: list[str], env: dict, log: Path) -> subprocess.Popen:
    with log.open("ab") as sink:
        return subprocess.Popen(argv, env=env, stdout=subprocess.DEVNULL, stderr=sink)


@dataclass(frozen=True)
class ProcessTools:
    """How a process is started and time is told; a test swaps in stand-ins."""

    launch: Launcher = _launch
    clock: Callable[[], float] = time.monotonic
    sleep: Callable[[float], None] = time.sleep


class CpuServer:
    """The process behind one processor model, and when it was last used."""

    def __init__(self, binary: str, model: CpuModel, log_dir: Path, probe: Probe,
                 tools: ProcessTools | None = None) -> None:
        tools = tools or ProcessTools()
        self.binary = binary
        self.model = model
        self.log = log_dir / f"cpu-{model.name}.log"
        self._probe = probe
        self._launch = tools.launch
        self._clock = tools.clock
        self._sleep = tools.sleep
        self._lock = threading.Lock()
        self._process: subprocess.Popen | None = None
        self.last_used = 0.0
        self.started_at = 0.0

    @property
    def running(self) -> bool:
        """Whether the process is up (it may still be loading)."""
        return self._process is not None and self._process.poll() is None

    def ensure_started(self) -> None:
        """Start the server if it is not running, and wait until it answers."""
        with self._lock:
            self.last_used = self._clock()
            if self.running:
                return
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
            self._process = self._launch(command(self.binary, self.model), env, self.log)
            self.started_at = self._clock()
            deadline = self.started_at + STARTUP_LIMIT_S
            while not self._probe(self.model.port):
                if self._process.poll() is not None:
                    self._process = None
                    raise RuntimeError(f"{self.model.name} stopped while starting; "
                                       f"see {self.log}")
                if self._clock() > deadline:
                    self._stop_locked()
                    raise TimeoutError(f"{self.model.name} did not start within "
                                       f"{STARTUP_LIMIT_S:.0f} s")
                self._sleep(POLL_S)

    def touch(self) -> None:
        """A request just used the model."""
        self.last_used = self._clock()

    def stop(self) -> None:
        """Stop the server, if it runs."""
        with self._lock:
            self._stop_locked()

    def stop_if_idle(self, busy: bool) -> bool:
        """Stop it when nothing used it for `idle_stop_s`; 0 keeps it running."""
        limit = self.model.idle_stop_s
        with self._lock:
            idle = self._clock() - self.last_used
            if busy or not limit or not self.running or idle < limit:
                return False
            self._stop_locked()
            return True

    def _stop_locked(self) -> None:
        process, self._process = self._process, None
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=STOP_LIMIT_S)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=STOP_LIMIT_S)
