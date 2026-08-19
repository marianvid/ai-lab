"""macOS host: plain child processes, Metal, unified memory.

There is no systemd and no sudo helper here. This application starts the
engine itself and holds the handle, which means inference stops when the
manager stops. That is acceptable on a laptop, and it is why Linux uses
systemd instead.

Two consequences worth remembering:

* `enabled` — starts at boot — is always False. Nothing supervises us.
* Apple silicon has no separate video memory. CPU and GPU share one pool, so
  there is no "VRAM freed" figure to watch. We report the engine process's
  resident memory against total system memory, and mark the reading
  `memory_kind="unified"` so the interface can say what it is showing instead
  of pretending it is a discrete card.
"""

from __future__ import annotations

import atexit
import signal
import subprocess
import tempfile
from pathlib import Path
from threading import RLock

from ..types import AcceleratorSnapshot, Capabilities, ProcessSpec, ProcessStatus
from .command import run, which

TERMINATE_GRACE_SECONDS = 20


class DarwinHost:
    """See `base.Host` for what each method promises."""

    def __init__(self) -> None:
        self._processes: dict[str, subprocess.Popen] = {}
        # Engine output goes to a file rather than being discarded: when a load
        # fails, the reason is in there and nowhere else.
        self._logs: dict[str, Path] = {}
        # The open file each engine writes to. Kept so it can be closed when
        # the process ends: leaving them open leaks a descriptor per start,
        # which on a laptop that swaps models all day eventually runs out.
        self._handles: dict[str, object] = {}
        self._lock = RLock()
        # Engines must not outlive the manager. Without this they would keep
        # running and holding their port, and the next manager would find a
        # stranger answering its health probe and believe its own load had
        # succeeded.
        atexit.register(self.stop_all)

    # -- capabilities ------------------------------------------------------

    def capabilities(self) -> Capabilities:
        """Only llama.cpp. vLLM needs CUDA and cannot run here at all."""
        engines = {"llamacpp"} if which("llama-server") else set()
        return Capabilities(
            supervisor="subprocess",
            engines=frozenset(engines),
            accelerator_kind="metal",
            can_configure_accelerator=False,
            operating_system="macOS",
        )

    # -- processes ---------------------------------------------------------

    def start(self, spec: ProcessSpec) -> None:
        with self._lock:
            if self._alive(spec.instance_id):
                raise RuntimeError(f"{spec.instance_id} is already running")
            import os

            log_path = Path(tempfile.gettempdir()) / f"ai-lab-{spec.instance_id}.log"
            self._logs[spec.instance_id] = log_path
            handle = open(log_path, "w")
            self._handles[spec.instance_id] = handle
            self._processes[spec.instance_id] = subprocess.Popen(
                spec.argv,
                env={**os.environ, **spec.env},
                stdout=handle,
                stderr=subprocess.STDOUT,
            )

    def stop(self, instance_id: str) -> None:
        """Ask politely, then insist.

        A large model can take a while to release memory, so SIGTERM is given
        a generous grace period before SIGKILL.
        """
        with self._lock:
            process = self._processes.get(instance_id)
            if process is None or process.poll() is not None:
                self._processes.pop(instance_id, None)
                return
            process.send_signal(signal.SIGTERM)
            try:
                process.wait(timeout=TERMINATE_GRACE_SECONDS)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
            self._processes.pop(instance_id, None)
            self._close_log(instance_id)

    def status(self, instance_id: str) -> ProcessStatus:
        with self._lock:
            process = self._processes.get(instance_id)
            running = process is not None and process.poll() is None
            return ProcessStatus(running=running,
                                 pid=process.pid if running else None,
                                 enabled=False)

    def _close_log(self, instance_id: str) -> None:
        handle = self._handles.pop(instance_id, None)
        if handle is not None:
            try:
                handle.close()
            except OSError:
                pass

    def logs(self, instance_id: str, lines: int = 15) -> list[str]:
        path = self._logs.get(instance_id)
        if path is None or not path.is_file():
            return []
        return path.read_text(errors="replace").splitlines()[-lines:]

    def stop_all(self) -> None:
        """Stop every engine. Runs when the manager exits."""
        with self._lock:
            identifiers = list(self._processes)
        for instance_id in identifiers:
            try:
                self.stop(instance_id)
            except Exception:
                pass

    def _alive(self, instance_id: str) -> bool:
        process = self._processes.get(instance_id)
        return process is not None and process.poll() is None

    # -- accelerator -------------------------------------------------------

    def accelerator(self, pid: int | None = None) -> AcceleratorSnapshot:
        with self._lock:
            pids = [process.pid for process in self._processes.values()
                    if process.poll() is None]
        return AcceleratorSnapshot(
            available=True,
            name=self._chip_name(),
            kind="metal",
            memory_kind="unified",
            memory_used_mb=self._resident_memory_mb(pids),
            memory_total_mb=self._total_memory_mb(),
            process_memory_mb=self._resident_memory_mb([pid]) if pid else 0.0,
        )

    @staticmethod
    def _resident_memory_mb(pids: list[int]) -> float:
        """Resident memory of the given processes, in megabytes.

        On a unified-memory machine this is the closest honest equivalent to
        "memory held by the model": the weights live in the process's own
        pages, so there is no separate pool to read.
        """
        pids = [pid for pid in pids if pid]
        if not pids:
            return 0.0
        result = run(["ps", "-o", "rss=", "-p", ",".join(str(pid) for pid in pids)], timeout=5)
        if not result.ok:
            return 0.0
        total_kb = sum(int(line) for line in result.stdout.split() if line.isdigit())
        return total_kb / 1024.0

    @staticmethod
    def _total_memory_mb() -> float:
        result = run(["sysctl", "-n", "hw.memsize"], timeout=5)
        try:
            return int(result.stdout.strip()) / (1024 * 1024)
        except ValueError:
            return 0.0

    @staticmethod
    def _chip_name() -> str:
        result = run(["sysctl", "-n", "machdep.cpu.brand_string"], timeout=5)
        return result.stdout.strip() or "Apple silicon"
