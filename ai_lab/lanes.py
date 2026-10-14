"""The gateway's lanes beside the card: one bundle, passed down to the front door.

The card's scheduler (`Gateway`) serves the models that need the GPU. Two
lanes never touch it and keep queues of their own:

- `providers`: subscription models run through their vendors' CLIs;
- `cpu`: models run on the processor, from RAM (embeddings).

Either may be absent, as it is in a test that needs only the card.
"""

from __future__ import annotations

from dataclasses import dataclass

from .cpu_models import CpuLane
from .providers import ProviderPool


@dataclass(frozen=True)
class SideLanes:
    """The lanes a request can take instead of a card lease."""

    providers: ProviderPool | None = None
    cpu: CpuLane | None = None

    def catalogue(self) -> list[dict]:
        """The models of both lanes, in the shape `GET /v1/models` lists."""
        rows: list[dict] = []
        if self.providers is not None:
            rows += self.providers.catalogue()
        if self.cpu is not None:
            rows += self.cpu.catalogue()
        return rows

    def describe(self, name: str) -> dict | None:
        """One model of either lane, or None when neither serves the name."""
        return next((row for row in self.catalogue() if row["id"] == name), None)

    def stop_all(self) -> None:
        """Stop what the lanes started themselves, when the manager quits."""
        if self.cpu is not None:
            self.cpu.stop_all()
