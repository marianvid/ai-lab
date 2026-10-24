"""What the rest of AI-Lab sees of the monitor: start it, ask it, stop it.

On a machine without an NVIDIA driver (the Mac), or with the monitor switched
off, there is no monitor; the page and the endpoint then say why, and
nothing else changes.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .alarms import AlarmBook, Condition
from .monitor import GpuHealthMonitor
from .nvml import LIBRARY, Nvml
from .output.report import weekly
from .settings import GpuHealthSettings
from .output.status import history, status
from .steps import Activity, Steps
from .store import Store

DB_NAME = "gpu-health.sqlite"


def log_channel(event: dict) -> None:
    """The "log" channel: one line in the service's log (journalctl -u ai-lab-manager)."""
    print(f"GPU health {event['level'].upper()} [{event['kind']}] {event['message']}", flush=True)


class GpuHealth:
    """The monitor when the card can be read; otherwise the reason it cannot."""

    def __init__(self, monitor: GpuHealthMonitor | None = None, reason: str = "",
                 path: Path | None = None) -> None:
        self.monitor = monitor
        self.reason = reason
        self.path = path

    @classmethod
    def start(cls, raw: Any, state_dir: Path, activity: Activity,
              load: Callable[[], Nvml] = Nvml.load) -> GpuHealth:
        """Check the settings, find the driver, start the thread."""
        settings = GpuHealthSettings.from_mapping(raw)
        if not settings.enabled:
            return cls(reason="Switched off in the configuration (gpu_health.enabled).")
        try:
            nvml = load()
        except OSError:
            return cls(reason=f"No NVIDIA driver on this machine ({LIBRARY} could not be loaded).")
        path = Path(state_dir) / DB_NAME
        store = Store(path)
        steps = Steps(settings, store, nvml, activity, AlarmBook(store, settings, log_channel))
        monitor = GpuHealthMonitor(steps, log_channel)
        monitor.start()
        return cls(monitor, path=path)

    def status(self) -> dict:
        """Everything the page shows."""
        if self.monitor is None:
            return {"available": False, "reason": self.reason}
        return status(self.monitor, time.time(), self.path)

    def history(self, hours: float) -> dict:
        """Temperature, power, fan and use over the last `hours`, for the chart."""
        return history(self._monitor().steps.store, time.time(), hours)

    def report(self, days: int) -> dict:
        """The report of the last `days` days, made now."""
        steps = self._monitor().steps
        return weekly(steps.store, steps.analysis, time.time(), days)

    def reset_baseline(self, note: str) -> dict:
        """Forget the references by hand, with a note saying why."""
        self._monitor().steps.analysis.reset_baselines(time.time(), note)
        return self.status()

    def test_alarm(self) -> dict:
        """Send one alarm through the channel, to see it arrive."""
        now = time.time()
        self._monitor().steps.book.event(now, Condition(
            f"test:{now:.0f}", "warning", "Test alarm from the GPU health page"))
        return self.status()

    def stop(self) -> None:
        """Stop the thread, if there is one, and close its file."""
        if self.monitor is not None:
            self.monitor.stop()
            self.monitor.steps.store.close()

    def _monitor(self) -> GpuHealthMonitor:
        if self.monitor is None:
            raise ValueError(f"GPU health is not available here: {self.reason}")
        return self.monitor
