"""Who else may hold card memory beside the card's own models, and gives it back.

The embeddings lane may borrow the card while it has room
(`docs/decisions/0004-embeddings-borrow-the-card.md`). The card's own models come
first, so whatever loads one asks its neighbours to step aside first and tells
them when it is over. `NoNeighbours` is the default: nobody borrows the card.
"""

from __future__ import annotations

import functools
from typing import Any, Protocol


class CardNeighbours(Protocol):
    """What a card load asks of whoever borrows the card."""

    def make_way(self) -> None:
        """Leave the card now; wait for requests already there to finish."""

    def settled(self) -> None:
        """The load is over, whether it worked or not."""

    def reclaimable_mb(self) -> float:
        """Card memory held now that would be given back on `make_way`."""


class NoNeighbours:
    """Nobody borrows the card."""

    def make_way(self) -> None:
        """Nothing to move."""

    def settled(self) -> None:
        """Nothing to bring back."""

    def reclaimable_mb(self) -> float:
        """Nothing held."""
        return 0.0


def guard_loads(service: Any, neighbours: CardNeighbours) -> None:
    """Make every `service.load` ask the neighbours to step aside first, and tell them after.

    `service` is the `InstanceService`: the one place every card load passes —
    the gateway, the Load button, and the restore after a restart. Wrapped here,
    at start-up, so the service itself need not know anyone borrows the card.
    """
    load = service.load

    @functools.wraps(load)
    def guarded(*args: Any, **kwargs: Any) -> Any:
        neighbours.make_way()
        try:
            return load(*args, **kwargs)
        finally:
            neighbours.settled()

    service.load = guarded
