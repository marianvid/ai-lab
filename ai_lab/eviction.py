"""Pure choice of which resident models must leave before another can load."""

from __future__ import annotations

from collections.abc import Callable


class EvictionPlanner:
    """Choose the cheapest victims using the gateway's last memory reading.

    No host calls occur here. The scheduler invokes this while holding its
    lock; reading the accelerator at that point would block every request.
    """

    def choose(self, wanted, loaded: list[dict], *,
               needs_mb: Callable[[object], float], free_mb: float,
               capacity_mb: float) -> list | None:
        needed = needs_mb(wanted)
        known = bool(needed and free_mb)
        if known and capacity_mb and needed > capacity_mb:
            return None

        idle = sorted((item for item in loaded if not item["in_flight"]),
                      key=lambda item: item["last_used"])
        busy = sorted((item for item in loaded if item["in_flight"]),
                      key=lambda item: item["last_used"])
        # A model reserving almost the entire card needs an empty card in
        # practice. Other engines' estimates are approximations of their
        # runtime allocations; adding them to an old free-memory reading can
        # claim enough space while CUDA still cannot load the new model.
        if known and capacity_mb and needed >= capacity_mb * 0.9:
            return [item["shape"] for item in idle + busy
                    if item["shape"] != wanted]
        if known and needed <= free_mb:
            return []
        victims = []
        for item in idle + busy:
            shape = item["shape"]
            if shape == wanted:
                continue
            victims.append(shape)
            if not known:
                continue
            free_mb += needs_mb(shape)
            if needed <= free_mb:
                return victims

        if known and needed > capacity_mb:
            return None
        return victims
