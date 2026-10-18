"""What is loaded on the accelerator right now, and moving between states.

A load is not an instant, it is a sequence, and each step is timed:

    unload:  stopping -> process_gone -> memory_released
    load:    starting -> process_up -> weights_loading -> ready

`process_up` means the binary is running. `ready` means it answered its own
health probe, which is the only trustworthy sign that the weights are actually
resident. The gap between the two is the interesting number, and it is what
fills the progress bar.

While a transition runs, accelerator memory is sampled several times a second
and every reading is published as an event: memory falls through an unload and
rises through a load. A swap is an unload followed by a load, reported as one
operation with a total, because that is the figure worth comparing between
models.

This module performs no I/O of its own. It is handed a host and an engine,
which is what makes the whole lifecycle testable on a laptop with no GPU.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from threading import RLock

from ..config.store import Instance
from ..engines.base import Engine, withheld
from ..events import EventBus
from ..hosts.base import Host
from .checks import MIB, StartChecks
from .diagnostics import RuntimeDiagnostics
from .state import (
    LOAD_SPAN,
    SWAP_LOAD_SPAN,
    SWAP_UNLOAD_SPAN,
    UNLOAD_SPAN,
    Operation,
    RuntimeProgress,
    Step,
)
from ..types import (
    AcceleratorSnapshot,
    ChangeEvent,
    ModelSet,
    Phase,
    ProcessSpec,
    ProcessStatus,
    RuntimeEvent,
)

__all__ = ["EngineDiedError", "LoadTimeoutError", "Operation", "Runtime"]

SAMPLE_INTERVAL_S = 0.2
# Starting a process takes seconds; loading weights can take minutes. They get
# separate limits so a process that never appears is not waited on for a
# quarter of an hour.
START_TIMEOUT_S = 60.0
LOAD_TIMEOUT_S = 900.0
UNLOAD_TIMEOUT_S = 120.0
# Memory is considered released once it has stopped falling for this many
# consecutive samples. Drivers free asynchronously, so the process exiting is
# not the same moment as the memory coming back.
SETTLED_SAMPLES = 3
# A sample this much lower than the one before still counts as memory holding still.
SETTLED_TOLERANCE_MB = 1.0


class LoadTimeoutError(RuntimeError):
    """A step of a load or unload took longer than its limit."""


class EngineDiedError(RuntimeError):
    """The engine process exited before it finished loading."""


@dataclass(frozen=True)
class _Wait:
    """What one wait is for: the phase shown meanwhile, its limit, and what it waits for."""

    phase: Phase
    timeout_s: float
    description: str


class Runtime:
    """Loading, unloading and swapping models, timed step by step and published as events.

    It holds its `StartChecks` (asked before every start) and owns every lock:
    one per instance, and `_guard` for the dictionaries below.
    """

    def __init__(self, host: Host, bus: EventBus,
                 sample_interval_s: float = SAMPLE_INTERVAL_S,
                 start_timeout_s: float = START_TIMEOUT_S,
                 load_timeout_s: float = LOAD_TIMEOUT_S) -> None:
        self.host = host
        self.checks = StartChecks(host)
        self.diagnostics = RuntimeDiagnostics(host)
        self.bus = bus
        self.sample_interval_s = sample_interval_s
        self.start_timeout_s = start_timeout_s
        self.load_timeout_s = load_timeout_s
        self._locks: dict[str, RLock] = {}
        self._guard = RLock()
        self._last: dict[str, Operation] = {}
        self._pids: dict[str, int | None] = {}
        self._progress: dict[str, RuntimeProgress] = {}
        # What each running instance was actually started with. Usually its
        # stored settings, but not always: a request may ask for a model with
        # a bigger context than the entry is configured for, and then the two
        # differ until it is unloaded. Remembered here because this is where
        # the launching happens, and reported so that the page does not show
        # settings the running model is not using.
        self._active: dict[str, dict] = {}

    # -- public operations -------------------------------------------------

    def load(self, instance: Instance, model: ModelSet, engine: Engine) -> Operation:
        """Start the engine with the model and wait until it answers; never raises."""
        with self._lock_for(instance.id):
            operation = Operation(instance.id, "load")
            self._begin(instance.id, LOAD_SPAN, target_mb=_megabytes(model))
            clock = _Clock()
            try:
                self._load(instance, model, engine, operation, clock)
                operation.ok = True
            except Exception as error:                      # reported, not raised
                self._fail(operation, clock, error)
            operation.total_ms = clock.elapsed_ms()
            return self._remember(operation)

    def unload(self, instance_id: str) -> Operation:
        """Stop the engine and wait until its memory is back; never raises."""
        with self._lock_for(instance_id):
            operation = Operation(instance_id, "unload")
            self._begin(instance_id, UNLOAD_SPAN,
                        baseline_mb=self._resident(instance_id))
            clock = _Clock()
            try:
                self._unload(instance_id, operation, clock)
                operation.ok = True
            except Exception as error:
                self._fail(operation, clock, error)
            operation.total_ms = clock.elapsed_ms()
            return self._remember(operation)

    def swap(self, instance: Instance, model: ModelSet, engine: Engine) -> Operation:
        """Unload then load, timed as one operation.

        Reported together because the question being asked is "how long to
        change model", not "how long to stop" and "how long to start".
        """
        with self._lock_for(instance.id):
            operation = Operation(instance.id, "swap")
            clock = _Clock()
            try:
                if self.host.status(instance.id).running:
                    self._begin(instance.id, SWAP_UNLOAD_SPAN,
                                baseline_mb=self._resident(instance.id))
                    self._unload(instance.id, operation, clock)
                self._begin(instance.id, SWAP_LOAD_SPAN, target_mb=_megabytes(model))
                self._load(instance, model, engine, operation, clock)
                operation.ok = True
            except Exception as error:
                self._fail(operation, clock, error)
            operation.total_ms = clock.elapsed_ms()
            return self._remember(operation)

    def status(self, instance: Instance, engine: Engine,
               process: ProcessStatus | None = None) -> dict:
        """One row of the model list.

        `process` is the supervisor's answer, when the caller has already got
        it. Reading it is the expensive part of drawing the whole list, and on
        systemd the answer for every instance comes back in one command — see
        `Host.statuses`. Left out, this asks for itself.
        """
        if process is None:
            process = self.host.status(instance.id)
        return {
            "id": instance.id,
            "engine": instance.engine,
            "model_id": instance.model_id,
            "port": instance.port,
            "params": instance.params,
            # What it is running with, when that is not what it is configured
            # with. Empty otherwise, so the page only mentions a difference
            # when there is one.
            "active_params": self.active_params(instance),
            "running": process.running,
            "pid": process.pid,
            "ready": engine.ready(instance.port) if process.running else False,
            "web_ui": bool(getattr(engine, "web_ui", lambda: None)()),
            # What the engine will not do with these settings, whatever the
            # weights can — pictures on mlx-lm, or on vLLM "text only". The
            # page takes those icons off the row.
            "withheld": sorted(withheld(engine, instance.params)),
            "last_operation": self.last(instance.id),
        }

    def active_params(self, instance: Instance) -> dict:
        """The settings the running process was started with, if they differ.

        A request can ask for a model with settings the entry is not configured
        for — a bigger context, usually — and the model is then reloaded with
        them without the stored configuration being touched. Somebody looking
        at the page would otherwise read the configured value and believe the
        running model was using it.
        """
        active = self._active.get(instance.id)
        if not active:
            return {}
        return {key: value for key, value in active.items()
                if instance.params.get(key) != value}

    def last(self, instance_id: str) -> dict | None:
        """The last load, unload or swap of an instance, or None."""
        operation = self._last.get(instance_id)
        return operation.json() if operation else None

    # -- the sequences -----------------------------------------------------

    def _load(self, instance: Instance, model: ModelSet, engine: Engine,
              operation: Operation, clock: _Clock) -> None:
        params = self.checks.split_if_it_will_not_fit(instance, model, engine)
        plan = engine.plan(model, instance.port, params)
        self._active[instance.id] = dict(params)
        self.checks.refuse_stranger(instance, engine)
        self.checks.refuse_if_it_cannot_fit(instance, model, plan)
        self._mark(operation, clock, Phase.STARTING, f"Starting {model.name}")
        self.host.start(ProcessSpec(instance.id, plan.argv, plan.env))

        self._await(lambda: self.host.status(instance.id).running, instance.id, clock,
                    _Wait(Phase.STARTING, self.start_timeout_s,
                          "waiting for the process to start"))
        # From here on the samples can be attributed to this process, which is
        # what makes the bar mean something when another model is resident.
        self._pids[instance.id] = self.host.status(instance.id).pid
        self._mark(operation, clock, Phase.PROCESS_UP, "Process running")

        self._await(lambda: engine.ready(instance.port), instance.id, clock,
                    _Wait(Phase.WEIGHTS_LOADING, self.load_timeout_s,
                          "waiting for the weights to load"),
                    alive=lambda: self.host.status(instance.id).running)
        self._mark(operation, clock, Phase.READY, f"{model.name} ready")

    def _unload(self, instance_id: str, operation: Operation, clock: _Clock) -> None:
        # Remember the pid before stopping, so the samples during the stop can
        # still be attributed to it as its memory falls away.
        self._pids.setdefault(instance_id, self.host.status(instance_id).pid)
        self._mark(operation, clock, Phase.STOPPING, "Stopping")
        self.host.stop(instance_id)

        self._await(lambda: not self.host.status(instance_id).running, instance_id, clock,
                    _Wait(Phase.STOPPING, UNLOAD_TIMEOUT_S, "waiting for the process to exit"))
        self._mark(operation, clock, Phase.PROCESS_GONE, "Process gone")

        self._await_settled(instance_id, clock, UNLOAD_TIMEOUT_S)
        self._pids.pop(instance_id, None)
        self._active.pop(instance_id, None)
        self._mark(operation, clock, Phase.MEMORY_RELEASED, "Memory released")

    # -- waiting, while publishing what the accelerator is doing ------------

    def _await(self, condition: Callable[[], bool], instance_id: str, clock: _Clock,
               wait: _Wait, alive: Callable[[], bool] | None = None) -> None:
        """Poll until the condition holds, publishing a reading each time.

        The readings are the progress bar: this is the only place that decides
        how often the accelerator is sampled.

        `alive` guards against waiting for something that has already died. An
        engine that cannot fit its model exits within a couple of seconds, and
        without this the manager would sit through the full timeout — fifteen
        minutes of "loading" for a failure that happened immediately.
        """
        deadline = time.monotonic() + wait.timeout_s
        while True:
            self._publish(instance_id, wait.phase, clock)
            if condition():
                return
            if alive is not None and not alive():
                raise EngineDiedError(self._why(instance_id))
            if time.monotonic() > deadline:
                detail = self._why(instance_id)
                raise LoadTimeoutError(
                    f"Timed out after {wait.timeout_s:.0f}s {wait.description}. {detail}")
            time.sleep(self.sample_interval_s)

    def _why(self, instance_id: str) -> str:
        return self.diagnostics.why(instance_id)

    def _await_settled(self, instance_id: str, clock: _Clock, timeout_s: float) -> None:
        """Wait for memory to stop falling.

        The process exiting and the driver returning its memory are separate
        moments, so an unload is only finished once the number holds still.
        """
        deadline = time.monotonic() + timeout_s
        previous, steady = None, 0
        while steady < SETTLED_SAMPLES:
            snapshot = self._publish(instance_id, Phase.MEMORY_RELEASED, clock)
            # The card total, not the process: by now the process is gone, and
            # what we are waiting for is the driver handing its memory back.
            used = snapshot.memory_used_mb
            holding = previous is not None and used >= previous - SETTLED_TOLERANCE_MB
            steady = steady + 1 if holding else 0
            previous = used
            if time.monotonic() > deadline:
                return          # good enough; do not fail an unload over this
            time.sleep(self.sample_interval_s)

    def _publish(self, instance_id: str, phase: Phase, clock: _Clock) -> AcceleratorSnapshot:
        snapshot = self.host.accelerator(pid=self._pids.get(instance_id))
        self.bus.publish(self._event(instance_id, phase, clock, snapshot))
        return snapshot

    def _event(self, instance_id: str, phase: Phase, clock: _Clock,
               snapshot: AcceleratorSnapshot, mark: str | None = None) -> RuntimeEvent:
        """One update: how far along, and how much memory, kept separate.

        The memory figures are per instance rather than per card, since with
        two models resident the card total says nothing about either. `mark`
        is the message of a step completed; None for a reading on the way.
        """
        completed, message = mark is not None, mark or ""
        bar = self._progress.get(instance_id) or RuntimeProgress()
        return RuntimeEvent(
            instance_id=instance_id, phase=phase, elapsed_ms=clock.elapsed_ms(),
            progress=round(bar.value(phase, snapshot.process_memory_mb, completed), 4),
            memory_used_mb=snapshot.process_memory_mb,
            memory_total_mb=snapshot.memory_total_mb,
            accelerator_used_mb=snapshot.memory_used_mb,
            message=message,
        )

    def _begin(self, instance_id: str, span: tuple[float, float],
               target_mb: float = 0.0, baseline_mb: float = 0.0) -> None:
        bar = RuntimeProgress()
        bar.span = span
        bar.target_mb = target_mb
        bar.baseline_mb = baseline_mb
        self._progress[instance_id] = bar

    def _resident(self, instance_id: str) -> float:
        """How much this instance holds right now — where an unload starts from."""
        pid = self._pids.get(instance_id) or self.host.status(instance_id).pid
        self._pids[instance_id] = pid
        return self.host.accelerator(pid=pid).process_memory_mb

    # -- bookkeeping -------------------------------------------------------

    def _mark(self, operation: Operation, clock: _Clock, phase: Phase, message: str) -> None:
        operation.steps.append(Step(phase, clock.elapsed_ms()))
        snapshot = self.host.accelerator(pid=self._pids.get(operation.instance_id))
        self.bus.publish(self._event(operation.instance_id, phase, clock, snapshot, message))

    def _fail(self, operation: Operation, clock: _Clock, error: Exception) -> None:
        operation.ok = False
        operation.error = str(error) or error.__class__.__name__
        self._mark(operation, clock, Phase.FAILED, operation.error)

    def _remember(self, operation: Operation) -> Operation:
        self._last[operation.instance_id] = operation
        # The phase events described the move; this says the result is in, so
        # anyone showing the list can ask for it again.
        self.bus.publish(ChangeEvent(topic="instances"))
        return operation

    def _lock_for(self, instance_id: str) -> RLock:
        """One lock per instance, so two tabs cannot start the same model twice."""
        with self._guard:
            return self._locks.setdefault(instance_id, RLock())


def _megabytes(model: ModelSet) -> float:
    """Where a load is heading: the weights have to arrive in full."""
    return model.size_bytes / MIB


class _Clock:
    """Milliseconds since an operation began."""

    def __init__(self) -> None:
        self._start = time.monotonic()

    def elapsed_ms(self) -> int:
        """How long since it began."""
        return int((time.monotonic() - self._start) * 1000)
