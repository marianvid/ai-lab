"""Immutable operation steps and mutable progress for process transitions."""

from __future__ import annotations

from dataclasses import dataclass, field

from .types import Phase


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


class RuntimeProgress:
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
