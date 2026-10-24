"""What the monitor does at each turn: read, store, check, analyse, tidy up.

Each step is separate so that one failing (NVML not answering, a full disk)
does not stop the others; the monitor runs them and counts the failures.
Everything that must survive a restart is kept in the store's `state` table.
"""

from __future__ import annotations

from collections.abc import Callable

from . import readings
from .alarms import AlarmBook, Condition
from .analysis.references import Analysis
from .analysis.checks import FastChecks, slow_conditions, slow_events
from .kernel.events import KernelEvents
from .nvml import Nvml
from .settings import GpuHealthSettings
from .store import Store
from .analysis.windows import IdleFinder, StableFinder

Activity = Callable[[], tuple[str, bool]]


class Steps:
    """The monitor's work, one method per kind of turn."""

    def __init__(self, settings: GpuHealthSettings, store: Store, nvml: Nvml,
                 activity: Activity, book: AlarmBook) -> None:
        self.settings, self.store, self.book = settings, store, book
        self._nvml, self._activity = nvml, activity
        self._opened = False
        self.missing: set[str] = set()
        self.last_ok: float | None = None
        self.started: float | None = None
        self._stable, self._idle = StableFinder(settings), IdleFinder(settings)
        self._checks = FastChecks(settings, store.state("fast_checks"))
        self.kernel = KernelEvents(settings, store.state("kernel_position"))
        self.analysis = Analysis(store, settings)

    def fast(self, now: float) -> None:
        """One 15-second reading: stored, fed to the window finders, checked."""
        self._open()
        try:
            sample = readings.fast(self._nvml, now, self.missing)
        except Exception:
            self._opened = False
            raise
        sample.model, sample.running = self._activity()
        self.store.add_sample(sample)
        self.last_ok = now
        row = sample.row()
        window, idle = self._stable.feed(row), self._idle.feed(row)
        if window is not None:
            self.store.add_window(window)
        if idle is not None:
            self.store.add_idle_window(idle["start"], idle["end"], idle["day"], idle["temp_median"])
        conditions, events = self._checks.check(row)
        self.book.conditions("fast", now, conditions)
        self._events(now, events)
        self.store.set_state("fast_checks", self._checks.state)

    def slow(self, now: float) -> None:
        """The hourly reading: memory errors, replaced rows, driver."""
        self._open()
        reading = readings.slow(self._nvml)
        values = {"ecc": reading.ecc, "rows": reading.rows, "driver": reading.driver}
        self._events(now, slow_events(values, self.store.state("slow_last")))
        self.book.conditions("slow", now, slow_conditions(values))
        self.store.add_slow_reading(now, reading.ecc, reading.rows, reading.driver)
        self.store.set_state("slow_last", values)

    def kernel_log(self, now: float) -> None:
        """New lines from the host helper, and whether it is still alive."""
        self._events(now, self.kernel.read())
        self.book.conditions("helper", now, self.kernel.silent(now))
        self.store.set_state("kernel_position", self.kernel.position)

    def silence(self, now: float) -> None:
        """NVML without an answer for longer than `nvml_silent_s`."""
        last = self.last_ok if self.last_ok is not None else self.started
        found = []
        if last is not None and now - last > self.settings.nvml_silent_s:
            found.append(Condition("nvml_silent", "alarm",
                                   f"No reading from the card for {now - last:.0f} s "
                                   "(NVML does not answer)", {"last_ok": self.last_ok}))
        self.book.conditions("nvml", now, found)

    def trend(self, now: float) -> None:
        """References and trend alarms."""
        self.book.conditions("trend", now, self.analysis.run(now))

    def fold(self, now: float) -> int:
        """Samples older than `raw_keep_days` folded into 5-minute rows."""
        return self.store.fold_old_samples(now - self.settings.raw_keep_days * 86400.0)

    def close(self) -> None:
        """Let go of NVML."""
        if self._opened:
            self._nvml.close()
            self._opened = False

    def _open(self) -> None:
        if not self._opened:
            self._nvml.open()
            self._opened = True

    def _events(self, now: float, events: list[Condition]) -> None:
        for event in events:
            self.book.event(now, event)
