"""The `cpu_models` section of the configuration, checked.

Shape::

    "cpu_models": {
      "binary": "/opt/ai/llama.cpp/build/bin/llama-server",
      "models": {
        "bge-m3": {"path": "/models/embeddings/bge-m3-FP16.gguf",
                   "pooling": "cls", "port": 8390}
      }
    }

Every model here runs on the processor, from RAM, never on the card.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import PurePosixPath

POOLINGS = ("cls", "mean", "last")
# Measured on CT 102 (12 cores) with bge-m3: 4 requests at once gave 3.0
# texts of about 300 tokens a second, against 2.4 in batches of 16 from one.
DEFAULT_PARALLEL = 4
DEFAULT_THREADS = 12
# Long enough for one run of a client, short enough not to hold RAM all day.
DEFAULT_IDLE_STOP_S = 1800.0
# Tokens shared by the `parallel` places: each text may have context / parallel
# tokens (2048 with the defaults). Measured with bge-m3: 4 × 8192 holds
# 8.5 GB of RAM and takes 13 s for one text of 2,800 tokens; 8192 holds about
# 2.6 GB, and sentences, titles and leads never come near 2048.
DEFAULT_CONTEXT = 8192
MIN_PORT, MAX_PORT = 1024, 65535


@dataclass(frozen=True)
class CpuModel:
    """One model on the processor: where it is, how it pools, its port and limits."""

    name: str
    path: str
    port: int
    pooling: str = "cls"
    parallel: int = DEFAULT_PARALLEL
    threads: int = DEFAULT_THREADS
    context: int = DEFAULT_CONTEXT
    idle_stop_s: float = DEFAULT_IDLE_STOP_S

    @classmethod
    def from_mapping(cls, name: str, raw: dict) -> CpuModel:
        """One entry of `cpu_models.models`."""
        if not isinstance(raw, dict):
            raise ValueError(f"cpu_models.models.{name} must be an object")
        path = raw.get("path")
        if not isinstance(path, str) or not PurePosixPath(path).is_absolute():
            raise ValueError(f"cpu_models.models.{name}.path must be an absolute path")
        pooling = raw.get("pooling", "cls")
        if pooling not in POOLINGS:
            raise ValueError(f"cpu_models.models.{name}.pooling must be one of {POOLINGS}")
        port = _whole(name, "port", raw.get("port"), MIN_PORT, MAX_PORT)
        return cls(name=name, path=path, port=port, pooling=pooling,
                   parallel=_whole(name, "parallel", raw.get("parallel", DEFAULT_PARALLEL), 1, 64),
                   threads=_whole(name, "threads", raw.get("threads", DEFAULT_THREADS), 1, 256),
                   context=_whole(name, "context", raw.get("context", DEFAULT_CONTEXT), 512, 131072),
                   idle_stop_s=_seconds(name, raw.get("idle_stop_s", DEFAULT_IDLE_STOP_S)))


@dataclass(frozen=True)
class CpuModelsConfig:
    """The llama.cpp server to run, and every model it runs on the processor."""

    binary: str = ""
    models: dict[str, CpuModel] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, raw: dict | None) -> CpuModelsConfig:
        """The whole section; empty or absent means no processor models."""
        if not raw:
            return cls()
        if not isinstance(raw, dict):
            raise ValueError("cpu_models must be an object")
        models = {name: CpuModel.from_mapping(name, item)
                  for name, item in (raw.get("models") or {}).items()}
        binary = raw.get("binary", "")
        if models and (not isinstance(binary, str) or not PurePosixPath(binary).is_absolute()):
            raise ValueError("cpu_models.binary must be an absolute path")
        ports = [model.port for model in models.values()]
        if len(ports) != len(set(ports)):
            raise ValueError("cpu_models: two models share a port")
        return cls(binary=binary, models=models)


def _whole(name: str, key: str, value, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ValueError(f"cpu_models.models.{name}.{key} must be a whole number "
                         f"from {low} to {high}")
    return value


def _seconds(name: str, value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise ValueError(f"cpu_models.models.{name}.idle_stop_s must be 0 or more "
                         "(0 keeps the model running)")
    return float(value)
