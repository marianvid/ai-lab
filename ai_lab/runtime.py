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
from dataclasses import dataclass, field
from threading import RLock

from .config import Instance
from .engines.base import Engine
from .events import EventBus
from .hosts.base import Host
from .types import (ChangeEvent, ModelSet, Phase, ProcessSpec, ProcessStatus,
                    RuntimeEvent)

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


@dataclass(frozen=True, slots=True)
class Step:
    phase: Phase
    elapsed_ms: int


@dataclass(slots=True)
class Operation:
    """The record of one load, unload or swap."""

    instance_id: str
    kind: str                       # "load", "unload" or "swap"
    ok: bool = False
    total_ms: int = 0
    steps: list[Step] = field(default_factory=list)
    error: str = ""

    def json(self) -> dict:
        return {
            "instance_id": self.instance_id,
            "kind": self.kind,
            "ok": self.ok,
            "total_ms": self.total_ms,
            "steps": [{"phase": step.phase.value, "elapsed_ms": step.elapsed_ms}
                      for step in self.steps],
            "error": self.error,
        }


# How the phases divide up the bar. A swap is an unload followed by a load, so
# its two halves share the range rather than each running 0 to 100.
LOAD_SPAN = (0.0, 1.0)
UNLOAD_SPAN = (0.0, 1.0)
SWAP_UNLOAD_SPAN = (0.0, 0.4)
SWAP_LOAD_SPAN = (0.4, 1.0)


class _Progress:
    """Turns phases and memory readings into a bar that runs 0 to 1.

    The long part of a load is the weights arriving, and that has a known
    destination — the size of the model — so it can be reported as a real
    fraction rather than a guess. The short phases either side get small fixed
    slices, because a bar that sits at zero and then jumps is worse than one
    that moves a little while a process starts.
    """

    def __init__(self) -> None:
        self.span = LOAD_SPAN
        self.target_mb = 0.0
        self.baseline_mb = 0.0
        self._last = 0.0

    def value(self, phase: Phase, process_mb: float, completed: bool = False) -> float:
        """`completed` marks the end of a phase rather than a sample inside it.

        The distinction matters at the end: while memory is still being handed
        back the bar should sit just short of full, and only the step that
        declares the phase finished may show 100%.
        """
        low, high = self.span
        fraction = 1.0 if completed and phase in (Phase.READY, Phase.MEMORY_RELEASED) \
            else self._fraction(phase, process_mb)
        self._last = low + (high - low) * fraction
        return self._last

    def _fraction(self, phase: Phase, process_mb: float) -> float:
        if phase is Phase.STARTING:
            return 0.02
        if phase is Phase.PROCESS_UP:
            return 0.05
        if phase is Phase.WEIGHTS_LOADING:
            share = process_mb / self.target_mb if self.target_mb else 0.0
            return 0.05 + 0.90 * min(1.0, max(0.0, share))
        if phase is Phase.READY:
            return 1.0
        if phase is Phase.STOPPING:
            if not self.baseline_mb:
                return 0.05
            gone = 1.0 - (process_mb / self.baseline_mb)
            return 0.05 + 0.60 * min(1.0, max(0.0, gone))
        if phase is Phase.PROCESS_GONE:
            return 0.75
        if phase is Phase.MEMORY_RELEASED:
            return 0.90
        return self._fraction_when_failed()

    def _fraction_when_failed(self) -> float:
        """A failure leaves the bar where it stopped rather than completing it."""
        low, high = self.span
        return (self._last - low) / (high - low) if high > low else 0.0


class Timeout(RuntimeError):
    pass


class EngineDied(RuntimeError):
    """The engine process exited before it finished loading."""


class Runtime:
    def __init__(self, host: Host, bus: EventBus,
                 sample_interval_s: float = SAMPLE_INTERVAL_S,
                 start_timeout_s: float = START_TIMEOUT_S,
                 load_timeout_s: float = LOAD_TIMEOUT_S) -> None:
        self.host = host
        self.bus = bus
        self.sample_interval_s = sample_interval_s
        self.start_timeout_s = start_timeout_s
        self.load_timeout_s = load_timeout_s
        self._locks: dict[str, RLock] = {}
        self._guard = RLock()
        self._last: dict[str, Operation] = {}
        self._pids: dict[str, int | None] = {}
        self._progress: dict[str, _Progress] = {}
        # What each running instance was actually started with. Usually its
        # stored settings, but not always: a request may ask for a model with
        # a bigger context than the entry is configured for, and then the two
        # differ until it is unloaded. Remembered here because this is where
        # the launching happens, and reported so that the page does not show
        # settings the running model is not using.
        self._active: dict[str, dict] = {}

    # -- public operations -------------------------------------------------

    def load(self, instance: Instance, model: ModelSet, engine: Engine) -> Operation:
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
               process: "ProcessStatus | None" = None) -> dict:
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
            "name": instance.name,
            "engine": instance.engine,
            "model_id": instance.model_id,
            "port": instance.port,
            "params": instance.params,
            # What it is running with, when that is not what it is configured
            # with. Empty otherwise, so the page only mentions a difference
            # when there is one.
            "active_params": self.active_params(instance),
            "running": process.running,
            "enabled": process.enabled,
            "pid": process.pid,
            "ready": engine.ready(instance.port) if process.running else False,
            "web_ui": bool(getattr(engine, "web_ui", lambda: None)()),
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
        operation = self._last.get(instance_id)
        return operation.json() if operation else None

    # -- the sequences -----------------------------------------------------

    def _load(self, instance: Instance, model: ModelSet, engine: Engine,
              operation: Operation, clock: "_Clock") -> None:
        plan = engine.plan(model, instance.port, instance.params)
        self._active[instance.id] = dict(instance.params)
        self._refuse_stranger(instance, engine)
        self._refuse_if_it_cannot_fit(instance, model, plan)
        self._mark(operation, clock, Phase.STARTING, f"Starting {model.name}")
        self.host.start(ProcessSpec(instance.id, plan.argv, plan.env))

        self._await(lambda: self.host.status(instance.id).running,
                    instance.id, clock, Phase.STARTING, self.start_timeout_s,
                    "waiting for the process to start")
        # From here on the samples can be attributed to this process, which is
        # what makes the bar mean something when another model is resident.
        self._pids[instance.id] = self.host.status(instance.id).pid
        self._mark(operation, clock, Phase.PROCESS_UP, "Process running")

        self._await(lambda: engine.ready(instance.port),
                    instance.id, clock, Phase.WEIGHTS_LOADING, self.load_timeout_s,
                    "waiting for the weights to load",
                    alive=lambda: self.host.status(instance.id).running)
        self._mark(operation, clock, Phase.READY, f"{model.name} ready")

    def _unload(self, instance_id: str, operation: Operation, clock: "_Clock") -> None:
        # Remember the pid before stopping, so the samples during the stop can
        # still be attributed to it as its memory falls away.
        self._pids.setdefault(instance_id, self.host.status(instance_id).pid)
        self._mark(operation, clock, Phase.STOPPING, "Stopping")
        self.host.stop(instance_id)

        self._await(lambda: not self.host.status(instance_id).running,
                    instance_id, clock, Phase.STOPPING, UNLOAD_TIMEOUT_S,
                    "waiting for the process to exit")
        self._mark(operation, clock, Phase.PROCESS_GONE, "Process gone")

        self._await_settled(instance_id, clock, UNLOAD_TIMEOUT_S)
        self._pids.pop(instance_id, None)
        self._active.pop(instance_id, None)
        self._mark(operation, clock, Phase.MEMORY_RELEASED, "Memory released")

    def _refuse_if_it_cannot_fit(self, instance: Instance, model: ModelSet,
                                 plan) -> None:
        """Refuse a model that is larger than the memory left on the card.

        Only a necessary condition, not a sufficient one: the weights are the
        floor, and the context cache sits on top of them, so a model that
        passes this check can still turn out not to fit. But when the weights
        alone exceed what is free, there is no need to start anything to find
        out — and the alternative is a confusing crash a few seconds later.

        A plan that deliberately leaves part of the model in system memory is
        exempt. There the weights are not all meant to be on the card, so
        comparing the whole file against free memory answers a question nobody
        asked. It is slow, and it is a choice — the setting that turns it on
        says how slow.
        """
        if getattr(plan, "splits_across_cpu", False):
            return
        snapshot = self.host.accelerator()
        if snapshot.memory_kind != "dedicated" or not snapshot.memory_total_mb:
            return
        free_mb = snapshot.memory_total_mb - snapshot.memory_used_mb
        needed_mb = model.size_bytes / (1024 * 1024)
        if needed_mb > free_mb:
            raise RuntimeError(
                f"{model.name} needs about {needed_mb / 1024:.1f} GB but only "
                f"{free_mb / 1024:.1f} GB is free on the card. Unload another "
                f"model, choose a smaller one, or set how many layers go on the "
                f"card and leave the rest in system memory.")

    def _refuse_stranger(self, instance: Instance, engine: Engine) -> None:
        """Refuse to start when something else already holds the port.

        Otherwise the readiness probe answers yes — from a process this
        manager did not start — and a load that never happened is reported as
        a success. The likely culprit is an engine left behind by a previous
        manager, or a second AI-Lab on the same machine.
        """
        if self.host.status(instance.id).running:
            return
        if engine.ready(instance.port):
            raise RuntimeError(
                f"Port {instance.port} is already answering, but this manager did "
                f"not start it. Stop the other process before loading {instance.id}.")

    # -- waiting, while publishing what the accelerator is doing ------------

    def _await(self, condition, instance_id: str, clock: "_Clock",
               phase: Phase, timeout_s: float, description: str,
               alive=None) -> None:
        """Poll until the condition holds, publishing a reading each time.

        The readings are the progress bar: this is the only place that decides
        how often the accelerator is sampled.

        `alive` guards against waiting for something that has already died. An
        engine that cannot fit its model exits within a couple of seconds, and
        without this the manager would sit through the full timeout — fifteen
        minutes of "loading" for a failure that happened immediately.
        """
        deadline = time.monotonic() + timeout_s
        while True:
            self._publish(instance_id, phase, clock)
            if condition():
                return
            if alive is not None and not alive():
                raise EngineDied(self._why(instance_id))
            if time.monotonic() > deadline:
                detail = self._why(instance_id)
                raise Timeout(f"Timed out after {timeout_s:.0f}s {description}. {detail}")
            time.sleep(self.sample_interval_s)

    def _why(self, instance_id: str) -> str:
        """Explain a death using the engine's own words.

        Its last error line names the actual cause — not enough memory, a
        corrupt file, an unsupported quantisation — which is what someone
        reading the message needs.
        """
        try:
            lines = self.host.logs(instance_id, lines=40)
        except Exception:
            lines = []
        # The supervisor's own summary lines come last and say only that
        # something exited. The useful sentence is the engine's, further up.
        lines = [line for line in lines if ".service:" not in line]
        errors = [line for line in lines
                  if any(marker in line.lower()
                         for marker in ("error", "failed", "cannot",
                                        "out of memory", "unable"))]
        if not lines:
            return ("The engine stopped while loading, and its output could not "
                    "be read. On Linux the manager needs to be in the "
                    "systemd-journal group to see it.")
        detail = (errors or lines)[-1]
        return f"The engine stopped while loading: {detail.strip()}"

    def _await_settled(self, instance_id: str, clock: "_Clock", timeout_s: float) -> None:
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
            steady = steady + 1 if previous is not None and used >= previous - 1.0 else 0
            previous = used
            if time.monotonic() > deadline:
                return          # good enough; do not fail an unload over this
            time.sleep(self.sample_interval_s)

    def _publish(self, instance_id: str, phase: Phase, clock: "_Clock"):
        snapshot = self.host.accelerator(pid=self._pids.get(instance_id))
        self.bus.publish(self._event(instance_id, phase, clock, snapshot))
        return snapshot

    def _event(self, instance_id: str, phase: Phase, clock: "_Clock", snapshot,
               message: str = "", completed: bool = False) -> RuntimeEvent:
        """One update: how far along, and how much memory, kept separate.

        The memory figures are per instance rather than per card, since with
        two models resident the card total says nothing about either.
        """
        bar = self._progress.get(instance_id) or _Progress()
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
        bar = _Progress()
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

    def _mark(self, operation: Operation, clock: "_Clock", phase: Phase, message: str) -> None:
        operation.steps.append(Step(phase, clock.elapsed_ms()))
        snapshot = self.host.accelerator(pid=self._pids.get(operation.instance_id))
        self.bus.publish(self._event(operation.instance_id, phase, clock,
                                     snapshot, message, completed=True))

    def _fail(self, operation: Operation, clock: "_Clock", error: Exception) -> None:
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
    return model.size_bytes / (1024 * 1024)


class _Clock:
    def __init__(self) -> None:
        self._start = time.monotonic()

    def elapsed_ms(self) -> int:
        return int((time.monotonic() - self._start) * 1000)
