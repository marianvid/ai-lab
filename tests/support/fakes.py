"""Stand-ins for the host and the engine.

These are the two interfaces that touch the outside world. Replacing them is
what lets the lifecycle be tested on a laptop with no GPU: the fake host
reports whatever memory curve a test asks for, so a load can be replayed
deterministically instead of waited for.
"""

from __future__ import annotations

from ai_lab.types import (AcceleratorSnapshot, Capabilities, ProcessSpec,
                          ProcessStatus)


class FakeHost:
    """Records what it was asked to do and returns a scripted memory curve."""

    def __init__(self, memory_curve: list[float] | None = None,
                 total_mb: float = 32000.0, other_models_mb: float = 0.0) -> None:
        # `other_models_mb` stands for a second model already resident, which
        # is what made the per-instance distinction necessary.
        self.other_models_mb = other_models_mb
        self.started: list[ProcessSpec] = []
        self.log_lines: list[str] = []
        self._pid = 4013
        self.stopped: list[str] = []
        self.running: set[str] = set()
        self.status_calls = 0
        self.total_mb = total_mb
        self._curve = list(memory_curve or [0.0])
        self._reading = 0

    def capabilities(self) -> Capabilities:
        return Capabilities(supervisor="fake", engines=frozenset({"llamacpp"}),
                            accelerator_kind="cuda", can_configure_accelerator=False,
                            operating_system="Test OS")

    def start(self, spec: ProcessSpec) -> None:
        self.started.append(spec)
        self.running.add(spec.instance_id)

    def stop(self, instance_id: str) -> None:
        self.stopped.append(instance_id)
        self.running.discard(instance_id)

    def logs(self, instance_id: str, lines: int = 15) -> list[str]:
        return self.log_lines

    def stop_all(self) -> None:
        self.running.clear()

    def status(self, instance_id: str) -> ProcessStatus:
        running = instance_id in self.running
        return ProcessStatus(running=running, pid=self._pid if running else None)

    def statuses(self, instance_ids: list[str]) -> dict[str, ProcessStatus]:
        """The batch question, which a fake has nothing to batch.

        It is here because the real hosts have it and a fake that answers a
        smaller set of questions than the thing it stands in for lets a caller
        pass its tests and fail on the machine. `self.status_calls` counts what
        was asked, for the tests that care that the list is drawn with one
        question rather than one per entry.
        """
        self.status_calls += 1
        return {identifier: self.status(identifier) for identifier in instance_ids}

    def accelerator(self, pid=None) -> AcceleratorSnapshot:
        used = self._curve[min(self._reading, len(self._curve) - 1)]
        self._reading += 1
        return AcceleratorSnapshot(
            available=True, name="Fake GPU", kind="cuda", memory_kind="dedicated",
            memory_used_mb=used + self.other_models_mb,
            memory_total_mb=self.total_mb,
            # Only the process that was asked about. Zero without a pid, which
            # is what a real card reports for a process that is not there.
            process_memory_mb=used if pid else 0.0,
        )


class FakeEngine:
    """Becomes ready after a fixed number of probes, once something is running.

    The host is passed in so the fake behaves like a real engine on a real
    port: nothing answers before a process exists. Without that, the runtime
    cannot tell "not started yet" from "a stranger already holds the port",
    which is a distinction it is required to make.
    """

    id = "fake"
    display_name = "Fake"

    def __init__(self, ready_after: int = 1, host=None,
                 splits_across_cpu: bool = False) -> None:
        self.ready_after = ready_after
        self.host = host
        self.splits_across_cpu = splits_across_cpu
        self.probes = 0

    def formats(self):
        from ai_lab.types import Format
        return frozenset({Format.GGUF})

    def params(self):
        return ()

    def plan(self, model, port, params):
        from ai_lab.engines.base import LaunchPlan
        return LaunchPlan(argv=["fake-server", model.entrypoint], env={},
                          health_path="/health",
                          splits_across_cpu=self.splits_across_cpu)

    def ready(self, port: int) -> bool:
        if self.host is not None and not self.host.running:
            return False
        self.probes += 1
        return self.probes >= self.ready_after
