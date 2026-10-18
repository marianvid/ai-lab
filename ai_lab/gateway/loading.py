"""How the gateway makes room on the card and puts a model on it.

`CardLoader`, an object the `Gateway` holds: which models have to go for
another to fit (decided from the last memory reading, under the scheduler's
lock), the numbers behind a refusal, and the switch itself — unload, wait for
the card to go quiet, load. The scheduler calls the switch with its lock
dropped.

It owns two things and nothing else touches them: the last memory reading
(`read_the_card` refreshes it, outside the scheduler's lock) and whether
something outside may have changed the card (`card_changed`, cleared by the
next sweep or adoption). The memory comes
in through the `CardMemory` interface, built by the gateway by default.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Callable
from typing import Protocol

from .. import budget
from .errors import CouldNotLoadError
from .eviction import EvictionPlanner
from .state import Shape
from .stats import GatewayCounters
from .card_operations import CardOperations


class CardMemory(Protocol):
    """What the loader asks of the machine's memory (see `resources.GatewayResources`)."""

    def needs_mb(self, shape, card_total_mb: float) -> float:
        """What this request shape needs, by its engine's estimate."""
        ...

    def card_total_mb(self) -> float:
        """The card's whole memory, or 0 when it cannot be read."""
        ...

    def capacity_mb(self, pools: dict) -> float:
        """What models may use of the pool, all told."""
        ...

    def free_mb(self, pools: dict) -> float:
        """What a card model may take now."""
        ...

    def read_pools(self) -> dict:
        """A fresh reading, pool by pool."""
        ...

    def wait_until_quiet(self, stopped: list) -> None:
        """Wait for the driver to hand back what the stopped models held."""
        ...


class CardLoader:
    """Room on the card, and loading into it.

    `scheduled()` is the scheduler's state report (what it believes is
    loaded); `engine_name_of(shape)` names a shape's engine for a refusal.
    """

    HISTORY = 50

    def __init__(self, operations: CardOperations, memory: CardMemory,
                 counters: GatewayCounters, scheduled: Callable[[], dict],
                 engine_name_of: Callable[[Shape | None], str]) -> None:
        self.operations = operations
        self._memory = memory
        self._counters = counters
        self._scheduled = scheduled
        self._engine_name_of = engine_name_of
        self._eviction = EvictionPlanner()
        # What the last card reading said, per pool. Refreshed around every
        # load, never from inside the scheduler's lock.
        self._pools: dict = {}
        # Whether anything outside may have changed what is on the card. Set at
        # startup and whenever a button on the page loads or unloads something.
        # The next switch sweeps: one expensive read after an outside change
        # rather than one on every request.
        self._resweep = True

    def card_changed(self) -> None:
        """Something outside may have changed the card: sweep and adopt again."""
        self._resweep = True

    def adopt_what_is_there(self, adopt: Callable[..., None]) -> None:
        """Tell the scheduler (`adopt`) what is already on the card, once.

        Only after something outside may have changed it — a manager that has
        just started, or a button on the page. Otherwise the scheduler is the
        authority and reading again would only be a chance to disagree with
        itself.

        Everything that is up and answering, however many. This used to insist
        on exactly one and give up otherwise — two engines up was not a card to
        adopt but a card to clear — and with a memory budget that is simply
        the ordinary state. The page reported "nothing loaded" while two models
        were serving.

        A model that is up but not yet answering is left for later, and the
        flag stays set so the question is asked again. Adopting it would send
        requests to a port with nothing behind it; giving up on it for good
        would leave it invisible until something else changed.
        """
        if not self._resweep:
            return
        # The expensive question, asked once after an outside change rather
        # than on every request.
        running = [item for item in self.operations.instances() if item["running"]]
        if any(not item["ready"] for item in running):
            return                      # still coming up: ask again next time
        self._resweep = False
        adopt(*[Shape.of(item["id"], {**item.get("params", {}),
                                      **item.get("active_params", {})})
                for item in running])

    def make_room(self, shape: Shape, loaded: list[dict]) -> list | None:
        """Choose victims from cached memory figures under the scheduler lock."""
        return self._eviction.choose(
            shape, loaded, needs_mb=self._needs_mb,
            free_mb=self._free_mb(), capacity_mb=self._capacity_mb())

    def why_it_does_not_fit(self, instance_id: str, asked_for: dict) -> dict:
        """The numbers behind a refusal, for a client to act on.

        Everything here is a fact about this machine right now. Nothing is
        remembered between requests: what a model took last time is knowledge
        that belongs to whatever is making the requests, not to this manager.
        """
        shape = Shape.of(instance_id, asked_for)
        pool = (self._pools.get(budget.CARD)
                or self._pools.get(budget.MACHINE) or {})
        return {
            "model": instance_id,
            "engine": self._engine_name_of(shape),
            "needed_mb": round(self._needs_mb(shape)),
            "capacity_mb": round(self._capacity_mb()),
            "available_mb": round(self._free_mb()),
            "pool": pool.get("name", ""),
            "loaded": [item.instance_id for item in self._scheduled_shapes()],
            "asked": dict(asked_for),
        }

    def _needs_mb(self, shape: Shape) -> float:
        return self._memory.needs_mb(shape, self._card_total_mb())

    def _card_total_mb(self) -> float:
        return self._memory.card_total_mb()

    def _capacity_mb(self) -> float:
        return self._memory.capacity_mb(self._pools)

    def _free_mb(self) -> float:
        return self._memory.free_mb(self._pools)

    def read_the_card(self) -> None:
        """Refresh the memory reading, keeping the last good one if the probe fails."""
        # Never called while holding the scheduler lock. Keep the last good
        # reading if the host probe fails during a model transition.
        with contextlib.suppress(Exception):
            self._pools = self._memory.read_pools()

    def put_on_card(self, shape: Shape, victims: tuple = ()) -> None:
        """Take those off, wait for the card to go quiet, then load this.

        `victims` is what the scheduler decided has to go. Empty means this
        fits beside what is already there and nothing is disturbed.

        A manager restart or a machine boot can leave engines up that nothing
        here knows about. Those are cleared too, on the first switch after such
        a change — see `card_changed` — because loading into whatever is left is
        how a model that is known to fit fails to fit.
        """
        started = time.perf_counter()
        unloaded = self._clear(victims)
        self._resweep = False
        self._memory.wait_until_quiet(unloaded)
        self.read_the_card()
        self._refuse_if_still_full(shape)

        operation = self.operations.load(shape.instance_id, shape.as_dict() or None)
        if not operation.ok:
            raise CouldNotLoadError(
                operation.error or f"{shape.instance_id} would not start")

        self.read_the_card()
        took = time.perf_counter() - started
        self._counters.switches += 1
        if unloaded:
            self._counters.evictions += len(unloaded)
        self._counters.switch_s += took
        self._counters.history.append({
            "at": time.time(), "loaded": shape.instance_id, "unloaded": unloaded,
            "took_s": round(took, 1), "load_ms": operation.total_ms,
        })
        del self._counters.history[:-self.HISTORY]

    def _clear(self, victims: tuple = ()) -> list[str]:
        """Unload what was chosen, and anything nothing here knows about.

        The second half is the point of sweeping rather than unloading by name:
        a manager restart can leave an engine running that the scheduler never
        adopted, and its memory is real whether or not this knows about it.
        """
        wanted = {shape.instance_id for shape in victims}
        known = {item.instance_id for item in self._scheduled_shapes()}
        stopped = []
        for instance in self.operations.instances():
            if not instance["running"]:
                continue
            # The chosen victims, and anything running that the scheduler
            # does not believe in. A stray is not in the budget arithmetic, so
            # its memory is unaccounted — and loading into what is left is how
            # a model that is known to fit fails to.
            if instance["id"] in wanted or instance["id"] not in known:
                self.operations.unload(instance["id"])
                stopped.append(instance["id"])
        return stopped

    def _scheduled_shapes(self) -> list:
        """What the scheduler believes is loaded."""
        return [item["shape"] for item in self._scheduled()["loaded"]]

    def _refuse_if_still_full(self, shape: Shape) -> None:
        """Say so before starting an engine that cannot fit.

        Asked only when the arithmetic is available. Where it is not, the
        machine was emptied instead and the memory reader has already
        insisted the card reached nothing.
        """
        needed = self._needs_mb(shape)
        free = self._free_mb()
        if not needed or not free or needed <= free:
            return
        raise CouldNotLoadError(
            f"{shape.instance_id} needs about {needed / 1024:.1f} GB and "
            f"{free / 1024:.1f} GB is free after unloading what could be "
            f"unloaded. Something outside AI-Lab is holding the card, or an "
            f"engine did not release its memory.")
