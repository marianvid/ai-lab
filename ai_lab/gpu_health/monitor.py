"""The monitor's thread: runs each step when it is due, and never stops on an error.

Turns:
- every `sample_s` (15 s): a fast reading; the host helper's new lines;
  whether NVML has been silent too long;
- every `slow_s` (1 h): the slow reading;
- every 5 minutes: references and trend;
- once a day: old samples folded into 5-minute rows;
- once a week: the weekly report.

A step that fails is counted and its error kept for the page; the thread
goes on with the next turn. Even a failure outside the steps is caught by
the loop and counted ("tick"), so the thread itself never ends on an error. The monitor only reads from the card (NVML
queries, no settings changed) and never opens it as a compute process, so it
does not appear in `nvidia-smi`'s process list.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

from .output.report import weekly
from .steps import Steps
from .analysis.windows import day_of

ANALYSIS_S = 300.0
WEEK_S = 7 * 86400.0


class GpuHealthMonitor:
    """Schedules the steps; `tick(now)` is one turn, the thread calls it in a loop."""

    def __init__(self, steps: Steps, send: Callable[[dict], None],
                 clock: Callable[[], float] = time.time) -> None:
        self.steps = steps
        self._send = send
        self._clock = clock
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._due: dict[str, float] = {}
        self.failures: dict[str, dict] = {}

    def start(self) -> None:
        """Run in the background until `stop`."""
        self.steps.started = self._clock()
        self._thread = threading.Thread(target=self._loop, name="gpu-health", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the thread and let go of NVML."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self.steps.close()

    @property
    def running(self) -> bool:
        """Whether the thread is alive."""
        return self._thread is not None and self._thread.is_alive()

    def _loop(self) -> None:
        while not self._stop.is_set():
            now = self._clock()
            try:
                self.tick(now)
            except Exception as error:  # noqa: BLE001 - the thread must outlive any failure
                self._failed("tick", now, error)
            self._stop.wait(self.steps.settings.sample_s)

    def tick(self, now: float) -> None:
        """One turn: every step that is due."""
        if self.steps.started is None:
            self.steps.started = now
        settings = self.steps.settings
        self._step("fast", now, settings.sample_s, self.steps.fast)
        self._step("kernel_log", now, settings.sample_s, self.steps.kernel_log)
        self._step("silence", now, settings.sample_s, self.steps.silence)
        self._step("slow", now, settings.slow_s, self.steps.slow)
        self._step("trend", now, ANALYSIS_S, self.steps.trend)
        self._step("daily", now, 0.0, self._daily)

    def _step(self, name: str, now: float, every: float, run: Callable[[float], object]) -> None:
        """Run `run(now)` if it is due; a failure is kept, never raised."""
        # A little early is still due: the loop sleeps exactly `sample_s`
        # after the work, so turns drift by the time the work took.
        if now < self._due.get(name, float("-inf")) - 1.0:
            return
        self._due[name] = now + every
        try:
            run(now)
        except Exception as error:  # noqa: BLE001 - the monitor must outlive any failure
            self._failed(name, now, error)
            return
        if name in self.failures:
            self.failures[name]["recovered"] = now

    def _failed(self, name: str, now: float, error: Exception) -> None:
        """Count a failure and keep its message for the page."""
        known = self.failures.setdefault(name, {"count": 0})
        known.update(count=known["count"] + 1, last=now, error=f"{type(error).__name__}: {error}")

    def _daily(self, now: float) -> None:
        """Once a day the fold; once a week the report."""
        store = self.steps.store
        if store.state("last_fold_day") != day_of(now):
            self._step("fold", now, 0.0, self.steps.fold)
            store.set_state("last_fold_day", day_of(now))
        last_report = store.state("last_report")
        if last_report is None:
            store.set_state("last_report", now)
        elif now - last_report >= WEEK_S:
            self._step("report", now, 0.0, self._report)

    def _report(self, now: float) -> None:
        report = weekly(self.steps.store, self.steps.analysis, now)
        event = {"ts": now, "kind": "report", "level": "info", "key": "weekly",
                 "message": report["text"], "detail": {**report, "sent": True}}
        self.steps.store.add_event(event)
        self.steps.store.set_state("last_report", now)
        self._send(event)
