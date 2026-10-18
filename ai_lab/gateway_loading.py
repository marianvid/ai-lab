"""How the gateway makes room on the card and puts a model on it.

A base class of `Gateway` (a mixin): which models have to go for another to
fit (decided from the last memory reading, under the scheduler's lock), the
numbers behind a refusal, and the switch itself — unload, wait for the card to
go quiet, load. The scheduler calls the switch with its lock dropped.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Callable

from . import budget
from .eviction import EvictionPlanner
from .gateway_errors import CouldNotLoadError
from .gateway_resources import GatewayResources
from .gateway_state import Shape
from .gateway_stats import GatewayCounters
from .operations import Operations
from .scheduler import Scheduler


class CardLoading:
    """Room on the card, and loading into it (see `Gateway`, which owns the fields)."""

    HISTORY = 50

    operations: Operations
    resources: GatewayResources
    eviction: EvictionPlanner
    scheduler: Scheduler
    counters: GatewayCounters
    _budget_pools: dict
    _resweep: bool
    # From `GatewayCatalogue`, the other base class of `Gateway`.
    _engine_name_of: Callable[[Shape | None], str]

    def _make_room(self, shape: Shape, loaded: list[dict]) -> list | None:
        """Choose victims from cached memory figures under the scheduler lock."""
        return self.eviction.choose(
            shape, loaded, needs_mb=self._needs_mb,
            free_mb=self._free_mb(), capacity_mb=self._capacity_mb())

    def _why_it_does_not_fit(self, instance_id: str, asked_for: dict) -> dict:
        """The numbers behind a refusal, for a client to act on.

        Everything here is a fact about this machine right now. Nothing is
        remembered between requests: what a model took last time is knowledge
        that belongs to whatever is making the requests, not to this manager.
        """
        shape = Shape.of(instance_id, asked_for)
        pool = (self._budget_pools.get(budget.CARD)
                or self._budget_pools.get(budget.MACHINE) or {})
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
        return self.resources.needs_mb(shape, self._card_total_mb())

    def _card_total_mb(self) -> float:
        return self.resources.card_total_mb()

    def _capacity_mb(self) -> float:
        return self.resources.capacity_mb(self._budget_pools)

    def _free_mb(self) -> float:
        return self.resources.free_mb(self._budget_pools)

    def _read_the_card(self) -> None:
        """Refresh the memory reading, keeping the last good one if the probe fails."""
        # Never called while holding the scheduler lock. Keep the last good
        # reading if the host probe fails during a model transition.
        with contextlib.suppress(Exception):
            self._budget_pools = self.resources.read_pools()

    def _put_on_card(self, shape: Shape, victims: tuple = ()) -> None:
        """Take those off, wait for the card to go quiet, then load this.

        `victims` is what the scheduler decided has to go. Empty means this
        fits beside what is already there and nothing is disturbed.

        A manager restart or a machine boot can leave engines up that nothing
        here knows about. Those are cleared too, on the first switch after such
        a change — see `_resweep` — because loading into whatever is left is
        how a model that is known to fit fails to fit.
        """
        started = time.perf_counter()
        unloaded = self._clear(victims)
        self._resweep = False
        self._wait_until_quiet(unloaded)
        self._read_the_card()
        self._refuse_if_still_full(shape)

        operation = self.operations.load(shape.instance_id, shape.as_dict() or None)
        if not operation.ok:
            raise CouldNotLoadError(
                operation.error or f"{shape.instance_id} would not start")

        self._read_the_card()
        took = time.perf_counter() - started
        self.counters.switches += 1
        if unloaded:
            self.counters.evictions += len(unloaded)
        self.counters.switch_s += took
        self.counters.history.append({
            "at": time.time(), "loaded": shape.instance_id, "unloaded": unloaded,
            "took_s": round(took, 1), "load_ms": operation.total_ms,
        })
        del self.counters.history[:-self.HISTORY]

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
        return [item["shape"] for item in self.scheduler.state()["loaded"]]

    def _wait_until_quiet(self, stopped: list) -> None:
        self.resources._wait_until_quiet(stopped)

    def _wait_for_release(self, deadline: float, keeping: bool) -> None:
        self.resources._wait_for_release(deadline, keeping)

    def _refuse_if_still_full(self, shape: Shape) -> None:
        """Say so before starting an engine that cannot fit.

        Asked only when the arithmetic is available. Where it is not, the
        machine was emptied instead and `_wait_for_release` has already
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

    def _loaded(self) -> str | None:
        """What is on the card, read rather than remembered.

        A remembered value goes stale the moment something outside the gateway
        unloads a model — and that happens on purpose: a person can force past
        a busy card from the page, and the gateway is deliberately not told.
        Reading it costs one call the caller is making anyway, and it cannot be
        wrong.

        One name, and the first one found. This is used where a single answer
        is wanted — the page asks the scheduler for the whole set. Anything
        running that the scheduler does not know about is swept off by the next
        switch, because its memory is real and unaccounted for.
        """
        for instance in self.operations.instances():
            if instance["running"]:
                return instance["id"]
        return None
