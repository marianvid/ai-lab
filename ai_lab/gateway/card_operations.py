"""What the gateway asks of the operations: its own narrow view (design review D1).

The gateway used to take the whole `Operations` façade and reach through it
into the host, the engine registry and the configuration store. It now
declares exactly what it uses. `Operations` satisfies it; a test hands in a
fake that does no more than this.
"""

from __future__ import annotations

from typing import Any, Protocol

from ..budget import Budget


class CardOperations(Protocol):
    """The entries, their engines, the card and moving models on and off it."""

    def configured(self) -> list[dict]:
        """Every entry, configuration only (the cheap question)."""
        ...

    def instances(self) -> list[dict]:
        """Every entry with what it is doing (the expensive question)."""
        ...

    def instance(self, instance_id: str) -> dict:
        """One entry, configuration only."""
        ...

    def effective_params(self, instance_id: str, settings: dict) -> dict:
        """The entry's settings with these laid over them, checked."""
        ...

    def model_for(self, instance_id: str) -> Any:
        """The model an entry points at."""
        ...

    def engine(self, engine_id: str) -> Any:
        """The engine with this id; KeyError when there is none."""
        ...

    def load(self, instance_id: str, settings: dict | None = None) -> Any:
        """Start this entry's model."""
        ...

    def unload(self, instance_id: str) -> Any:
        """Stop this entry."""
        ...

    def accelerator(self) -> Any:
        """One reading of the card (an `AcceleratorSnapshot`)."""
        ...

    def machine_budget(self, card: Any = None) -> Budget:
        """What the machine can offer models now, pool by pool (`card`: a reading already taken)."""
        ...
