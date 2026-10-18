"""How a music engine is configured: its model files and its sampling settings.

Small value objects instead of long constructor argument lists. Each knows
what makes it valid and says so with the engine's name in the error.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# The ranges the manager accepts for sampling settings.
MAX_TOP_K = 1000
MAX_TEMPERATURE = 5.0
MAX_GUIDANCE = 10.0


@dataclass(frozen=True)
class Checkpoint:
    """A model bundle and the checkpoint inside it that the instance names.

    `checkpoint_subdir` is a single folder name inside `bundle_root`, and it
    must resolve to `checkpoint_path`: the bundle must point at exactly the
    checkpoint the manager configured, not at whatever it ships with.
    """

    bundle_root: Path
    checkpoint_path: Path
    checkpoint_subdir: str

    def check(self, engine: str) -> None:
        """Refuse a bundle or checkpoint that is absent or does not match."""
        if not self.bundle_root.is_dir() or not self.checkpoint_path.is_dir():
            raise ValueError(f"{engine} bundle or checkpoint is absent")
        inside = self.bundle_root / self.checkpoint_subdir
        if (Path(self.checkpoint_subdir).name != self.checkpoint_subdir
                or inside.resolve() != self.checkpoint_path.resolve()):
            raise ValueError(f"{engine} bundle does not point to the configured checkpoint")


@dataclass(frozen=True)
class Sampling:
    """Token sampling for an autoregressive music model: top-k, temperature, guidance."""

    topk: int
    temperature: float
    cfg_scale: float

    def check(self, engine: str) -> None:
        """Refuse settings outside the ranges the models are run with."""
        if (not 1 <= self.topk <= MAX_TOP_K or not 0 <= self.temperature <= MAX_TEMPERATURE
                or not 0 < self.cfg_scale <= MAX_GUIDANCE):
            raise ValueError(f"{engine} inference settings are invalid")
