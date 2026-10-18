"""Accelerator budget readings and per-engine memory estimates for the gateway."""

from __future__ import annotations

import time

from .. import budget
from .card_sharing import CardNeighbours, NoNeighbours
from .errors import CouldNotLoadError


class GatewayResources:
    """What the card holds and what a model needs, read for the scheduler."""

    def __init__(self, operations, quiet_mb: float, quiet_timeout_s: float,
                 poll_s: float, neighbours: CardNeighbours | None = None) -> None:
        """`neighbours` borrow the card beside its models (the embeddings lane), built first."""
        self.operations = operations
        self.quiet_mb = quiet_mb
        self.quiet_timeout_s = quiet_timeout_s
        self.poll_s = poll_s
        self._neighbours: CardNeighbours = neighbours or NoNeighbours()

    def needs_mb(self, shape, card_total_mb: float) -> float:
        """Ask the selected engine how much this request shape needs."""
        try:
            instance = self.operations.instance(shape.instance_id)
            engine = self.operations.engines.get(instance["engine"])
            model = self.operations.model_for(shape.instance_id)
            params = {**instance["params"], **shape.as_dict()}
            return float(engine.needs_mb(model, params, card_total_mb))
        except Exception:
            return 0.0

    def card_total_mb(self) -> float:
        """The card's whole memory, or 0 when it cannot be read."""
        try:
            return float(self.operations.host.accelerator().memory_total_mb)
        except Exception:
            return 0.0

    @staticmethod
    def pool(pools: dict) -> dict:
        """The pool models load into: the card's, else the machine's."""
        return pools.get(budget.CARD) or pools.get(budget.MACHINE) or {}

    def capacity_mb(self, pools: dict) -> float:
        """What models may use of that pool, all told."""
        return float(self.pool(pools).get("for_models_mb", 0.0))

    def free_mb(self, pools: dict) -> float:
        """What a card model may take: what is free, plus what a neighbour gives back."""
        available = float(self.pool(pools).get("available_mb", 0.0))
        return available + self._neighbours.reclaimable_mb()

    def read_pools(self) -> dict:
        """Read outside the scheduler lock; a failed probe keeps old readings."""
        found = budget.of(self.operations.host, self.operations.reserve_mb())
        return {pool.name: pool.json() for pool in found.pools}

    def wait_until_quiet(self, stopped: list) -> None:
        """Wait for the driver to hand back what those models held.

        A process exits before its VRAM is released. Loading in that gap fails
        with a message about the model being too large, which sends whoever
        reads it looking in the wrong place entirely.

        Waited for by name, not by watching the total fall to nothing: with a
        memory budget, other models stay loaded and their memory is not coming
        back. What has to be gone is the processes that were stopped, and that
        is a question with an exact answer.

        Neighbours borrowing the card leave it first, whatever was stopped: the
        card is read right after this, and must be read without them.
        """
        self._neighbours.make_way()
        if not stopped:
            return
        snapshot = self.operations.host.accelerator()
        if snapshot.memory_kind != "dedicated":
            return                      # unified memory: nothing to wait for
        deadline = time.perf_counter() + self.quiet_timeout_s
        while True:
            running = {item["id"] for item in self.operations.instances()
                       if item["running"]}
            still_up = [name for name in stopped if name in running]
            if not still_up:
                self._wait_for_release(deadline, keeping=bool(running))
                return
            if time.perf_counter() > deadline:
                raise CouldNotLoadError(
                    f"{', '.join(still_up)} did not exit within "
                    f"{self.quiet_timeout_s:.0f} seconds. The card is still "
                    f"holding what it was using.")
            time.sleep(self.poll_s)

    def _wait_for_release(self, deadline: float, keeping: bool) -> None:
        """Wait for the driver to hand back what those processes held.

        A process exits before its memory does, and loading into that gap fails
        with a message about the model being too large, which sends whoever
        reads it looking in the wrong place entirely.

        What to wait *for* depends on what is left. With nothing meant to be
        loaded, the card should reach nothing, and anything else is a fault
        worth naming — something outside this manager holding it, or an engine
        that did not release. With other models still loaded, the card
        legitimately holds their memory and there is no figure to wait for; the
        question becomes whether there is room, which `_refuse_if_still_full`
        asks precisely once the reading is fresh.
        """
        if keeping:
            return
        while True:
            used = self.operations.host.accelerator().memory_used_mb
            if used <= self.quiet_mb:
                return
            if time.perf_counter() > deadline:
                raise CouldNotLoadError(
                    f"The card still holds {used:.0f} MB "
                    f"{self.quiet_timeout_s:.0f} seconds after everything was "
                    f"unloaded. Something outside AI-Lab is using it, or an "
                    f"engine did not exit.")
            time.sleep(self.poll_s)
