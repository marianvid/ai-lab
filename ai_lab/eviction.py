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
        """The shapes to unload, oldest idle first; None when it can never fit."""
        needed = needs_mb(wanted)
        known = bool(needed and free_mb)
        if known and _over(needed, capacity_mb):
            return None
        others = [item["shape"] for item in _oldest_idle_first(loaded)
                  if item["shape"] != wanted]
        # Without a memory reading there is nothing to count with, so every
        # other model leaves. A model reserving almost the entire card needs
        # an empty card in practice: other engines' estimates approximate
        # their runtime allocations, and adding them to an old free-memory
        # reading can claim enough space while CUDA still cannot load it.
        if not known or _over(needed, capacity_mb * 0.9, inclusive=True):
            return others
        if needed <= free_mb:
            return []
        return _until_it_fits(others, needed, free_mb, needs_mb, capacity_mb)


def _over(needed: float, limit: float, *, inclusive: bool = False) -> bool:
    """Whether `needed` passes a known limit; a zero limit means unknown."""
    if not limit:
        return False
    return needed >= limit if inclusive else needed > limit


def _oldest_idle_first(loaded: list[dict]) -> list[dict]:
    """Idle models before busy ones, each group least recently used first."""
    return sorted(loaded, key=lambda item: (bool(item["in_flight"]),
                                            item["last_used"]))


def _until_it_fits(others: list, needed: float, free_mb: float,
                   needs_mb: Callable[[object], float],
                   capacity_mb: float) -> list | None:
    victims = []
    for shape in others:
        victims.append(shape)
        free_mb += needs_mb(shape)
        if needed <= free_mb:
            return victims
    return None if needed > capacity_mb else victims
