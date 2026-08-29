"""macOS host: plain child processes, Metal, unified memory.

There is no systemd and no sudo helper here. This application starts the
engine itself and holds the handle, which means inference stops when the
manager stops. That is acceptable on a laptop, and it is why Linux uses
systemd instead.

Two consequences worth remembering:

* Apple silicon has no separate video memory. CPU and GPU share one pool, so
  there is no "VRAM freed" figure to watch. We report the engine process's
  resident memory against total system memory, and mark the reading
  `memory_kind="unified"` so the interface can say what it is showing instead
  of pretending it is a discrete card.
"""

from __future__ import annotations

import atexit
import re
import signal
import subprocess
import tempfile
from pathlib import Path
from threading import RLock

from ..types import AcceleratorSnapshot, Capabilities, ProcessSpec, ProcessStatus
from .command import run, which

TERMINATE_GRACE_SECONDS = 20


# What counts as memory in use, from `vm_stat`. Deliberately not the whole of
# it: inactive and speculative pages are reclaimed under pressure, so counting
# them would report a healthy machine as nearly full and refuse a model that
# would have fitted.
USED_PAGES = ("Pages active", "Pages wired down", "Pages occupied by compressor")

class DarwinHost:
    """See `base.Host` for what each method promises."""

    def __init__(self, llamacpp_binary: str | None = None,
                 mlxwhisper_binary: str | None = None,
                 onnx_binary: str | None = None,
                 pyannote_binary: str | None = None,
                 paddleocr_binary: str | None = None,
                 comfyui_binary: str | None = None,
                 comfyui_main: str | None = None) -> None:
        self.engine_binaries = {
            "llamacpp": llamacpp_binary,
            "mlxwhisper": mlxwhisper_binary,
            "onnx": onnx_binary,
            "pyannote": pyannote_binary,
            "paddleocr": paddleocr_binary,
            "comfyui": comfyui_binary,
        }
        self.comfyui_main = comfyui_main
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
        """Portable and Metal-native engines; CUDA-only engines are absent."""
        supported = frozenset({"llamacpp", "mlxwhisper", "onnx", "pyannote",
                               "paddleocr", "comfyui"})
        engines = {key for key, binary in self.engine_binaries.items()
                   if self._installed(binary)}
        if "llamacpp" not in engines and which("llama-server"):
            engines.add("llamacpp")
        if "comfyui" in engines and not self._installed(self.comfyui_main):
            engines.remove("comfyui")
        return Capabilities(
            supervisor="subprocess",
            engines=frozenset(engines),
            accelerator_kind="metal",
            can_configure_accelerator=False,
            operating_system="macOS",
            supported_engines=supported,
        )

    @staticmethod
    def _installed(program: str | None) -> bool:
        if not program:
            return False
        return Path(program).is_file() or bool(which(program))

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

    def system_memory(self) -> tuple[float, float]:
        """What the whole machine is using, and how much it has.

        This used to return nothing, on the reasoning that Apple silicon shares
        one pool so the accelerator reading already covered it. That reasoning
        was wrong for the question that matters: the accelerator reading is
        what *our own engines* hold, and says nothing about the browser, the
        editor and everything else on a machine somebody is also working on.
        Deciding whether another model fits needs the second number.

        Counted conservatively — used is what cannot be taken away without
        someone noticing: memory in active use, memory the kernel has pinned,
        and memory it has already compressed to make room. Inactive and
        speculative pages are left out of "used" because the kernel will
        reclaim them under pressure, and counting them would report a machine
        that is behaving perfectly as nearly full.

        Being wrong optimistically here is the expensive direction: it means
        starting a model that does not fit, on a machine already holding one
        that was working.
        """
        try:
            total = float(run(["sysctl", "-n", "hw.memsize"], timeout=5).stdout)
        except (ValueError, AttributeError, TypeError):
            return 0.0, 0.0
        counts = self._pages()
        if not counts or not total:
            return 0.0, 0.0
        used = sum(counts.get(name, 0.0) for name in USED_PAGES)
        return used, total / (1024 * 1024)

    @staticmethod
    def _pages() -> dict[str, float]:
        """`vm_stat` in megabytes. Empty when it cannot be read."""
        result = run(["vm_stat"], timeout=5)
        if not result.ok or not result.stdout.strip():
            return {}
        lines = result.stdout.splitlines()
        size = re.search(r"page size of (\d+)", lines[0] if lines else "")
        if not size:
            return {}
        page_mb = int(size.group(1)) / (1024 * 1024)
        counts = {}
        for line in lines[1:]:
            name, _, rest = line.partition(":")
            digits = re.sub(r"\D", "", rest)
            if digits:
                counts[name.strip()] = int(digits) * page_mb
        return counts

    def state_dir(self) -> Path:
        """Beside the logs, under the user's own Library.

        Per user, not per machine: the manager here runs as whoever started it,
        and two accounts on one Mac are two installations.
        """
        directory = Path.home() / "Library" / "Application Support" / "AI-Lab"
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def statuses(self, instance_ids: list[str]) -> dict[str, ProcessStatus]:
        """One at a time, because here that costs nothing.

        This host owns the processes and keeps them in a dictionary, so asking
        about one is a lookup rather than a command. Only the Linux host, where
        every answer is a call to systemd, has anything to save by batching.
        """
        return {identifier: self.status(identifier) for identifier in instance_ids}

    def status(self, instance_id: str) -> ProcessStatus:
        with self._lock:
            process = self._processes.get(instance_id)
            running = process is not None and process.poll() is None
            return ProcessStatus(running=running,
                                 pid=process.pid if running else None)

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
