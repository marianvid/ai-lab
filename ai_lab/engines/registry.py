"""Which engines exist, and which are usable on this machine.

An engine is *known* if there is a file for it. It is *available* if the host
reports its binary is present. vLLM is known everywhere and available only
where CUDA is, so the interface can show it greyed out with a reason instead
of leaving the user wondering where it went.

The registry is an object rather than module state because engines are
configured: a machine can easily hold two builds of llama.cpp — one from a
package manager and one you compiled yourself — and resolving from PATH
silently picks whichever comes first. That may not be the one you tuned.
"""

from __future__ import annotations

from dataclasses import asdict

from ..types import Capabilities, Task
from .base import Engine
from .llamacpp import LlamaCppEngine
from .nemo import NemoEngine
from .onnx import OnnxEngine
from .pyannote import PyannoteEngine
from .vllm import VllmEngine


def build(settings: dict | None = None) -> dict[str, Engine]:
    """Create the engines, applying any per-engine configuration.

    `settings` is the `engines` section of config.json, for example
    `{"llamacpp": {"binary": "/opt/ai/llama.cpp/build/bin/llama-server"}}`.
    An engine with nothing configured falls back to looking on PATH.
    """
    settings = settings or {}
    return {
        LlamaCppEngine.id: LlamaCppEngine(
            binary=settings.get(LlamaCppEngine.id, {}).get("binary")),
        VllmEngine.id: VllmEngine(
            binary=settings.get(VllmEngine.id, {}).get("binary")),
        NemoEngine.id: NemoEngine(
            binary=settings.get(NemoEngine.id, {}).get("binary"),
            server=settings.get(NemoEngine.id, {}).get("server")),
        OnnxEngine.id: OnnxEngine(
            binary=settings.get(OnnxEngine.id, {}).get("binary"),
            server=settings.get(OnnxEngine.id, {}).get("server")),
        PyannoteEngine.id: PyannoteEngine(
            binary=settings.get(PyannoteEngine.id, {}).get("binary"),
            server=settings.get(PyannoteEngine.id, {}).get("server")),
    }


class Registry:
    """The engines this installation uses."""

    def __init__(self, settings: dict | None = None) -> None:
        self._engines = build(settings)

    def get(self, engine_id: str) -> Engine:
        engine = self._engines.get(engine_id)
        if engine is None:
            raise KeyError(f"Unknown engine: {engine_id}")
        return engine

    def known(self) -> dict[str, Engine]:
        return dict(self._engines)

    def available(self, capabilities: Capabilities) -> dict[str, Engine]:
        return {key: engine for key, engine in self._engines.items()
                if key in capabilities.engines}

    def describe(self, capabilities: Capabilities) -> list[dict]:
        """What the interface needs to draw the engine list.

        Unavailable engines are included, with the reason, rather than omitted.
        The binary path is reported too, so it is obvious which build is in use
        when two are installed.
        """
        rows = []
        for key, engine in self._engines.items():
            usable = key in capabilities.engines
            rows.append({
                "id": key,
                "name": engine.display_name,
                "available": usable,
                "reason": "" if usable else _reason(key, capabilities),
                "binary": getattr(engine, "binary", ""),
                "formats": sorted(item.value for item in engine.formats()),
                "tasks": sorted(item.value for item in engine.tasks()),
                "params": [asdict(spec) for spec in engine.params()],
                "task_params": {
                    task.value: [asdict(spec) for spec in engine.params(task)]
                    for task in engine.tasks()
                },
            })
        return rows


def _reason(engine_id: str, capabilities: Capabilities) -> str:
    if engine_id == "vllm" and capabilities.accelerator_kind != "cuda":
        return "Requires an NVIDIA GPU"
    if engine_id == "nemo" and capabilities.accelerator_kind != "cuda":
        return "Requires an NVIDIA GPU"
    if engine_id in ("vllm", "nemo", "onnx", "pyannote"):
        return "Not installed"
    return "Binary not found"
