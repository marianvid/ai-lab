"""Models that run on the processor, from RAM: embeddings, served through the gateway.

See `docs/decisions/0003-processor-models.md` for why they are a lane of their
own beside the card's scheduler, and `lane.py` for how a request goes.
"""

from .lane import EMBEDDINGS_PATH, CpuLane
from .settings import CpuModel, CpuModelsConfig

__all__ = ["EMBEDDINGS_PATH", "CpuLane", "CpuModel", "CpuModelsConfig"]
