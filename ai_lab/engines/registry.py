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

import importlib
import inspect
import pkgutil
from dataclasses import asdict

from ..types import Capabilities
from .base import Engine

# Modules of this package that hold no engine.
_NOT_ENGINES = frozenset({"base", "isolated", "probe", "registry", "text_launch"})


def engine_classes() -> dict[str, type]:
    """Every engine this package defines, by id, found by looking.

    An engine registers itself by being a module of this package with a class
    that has an `id`, a `display_name` and a `plan` method — the shape of
    `base.Engine`. Adding one touches only its own file; nothing here lists
    them by name. Ordered by module name, so the list is the same each time.
    """
    found: dict[str, type] = {}
    package = importlib.import_module(__package__ or "ai_lab.engines")
    for module_info in sorted(pkgutil.iter_modules(package.__path__),
                              key=lambda item: item.name):
        if module_info.name in _NOT_ENGINES or module_info.name.startswith("_"):
            continue
        module = importlib.import_module(f"{package.__name__}.{module_info.name}")
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if cls.__module__ == module.__name__ and _is_engine(cls):
                found[getattr(cls, "id", "")] = cls
    return found


def _is_engine(cls: type) -> bool:
    """Whether a class has the shape of an engine (see `base.Engine`)."""
    return (isinstance(getattr(cls, "id", None), str)
            and isinstance(getattr(cls, "display_name", None), str)
            and callable(getattr(cls, "plan", None)))


def _construct(cls: type, settings: dict) -> Engine:
    """One engine, given the settings its constructor takes.

    An engine's section of the configuration can hold more than the engine
    reads — `source` and `update`, for example, belong to the builds and
    installs that maintain it — so only the names its constructor declares
    are passed. An engine with nothing configured falls back to its defaults
    (usually: look on PATH). An engine whose settings are grouped (more than
    a constructor should take one by one) reads its section itself, with a
    `from_settings` class method.
    """
    make = getattr(cls, "from_settings", None)
    if make is not None:
        return make(settings)
    accepted = inspect.signature(cls).parameters
    return cls(**{key: value for key, value in settings.items() if key in accepted})


def build(settings: dict | None = None) -> dict[str, Engine]:
    """Create the engines, applying any per-engine configuration.

    `settings` is the `engines` section of config.json, for example
    `{"llamacpp": {"binary": "/opt/ai/llama.cpp/build/bin/llama-server"}}`.
    """
    settings = settings or {}
    return {engine_id: _construct(cls, settings.get(engine_id) or {})
            for engine_id, cls in engine_classes().items()}


class Registry:
    """The engines this installation uses."""

    def __init__(self, settings: dict | None = None) -> None:
        self._engines = build(settings)

    def get(self, engine_id: str) -> Engine:
        """The engine with this id; KeyError when there is none."""
        engine = self._engines.get(engine_id)
        if engine is None:
            raise KeyError(f"Unknown engine: {engine_id}")
        return engine

    def known(self) -> dict[str, Engine]:
        """Every engine, whether or not this machine can run it."""
        return dict(self._engines)

    def available(self, capabilities: Capabilities) -> dict[str, Engine]:
        """The engines this machine can run."""
        return {key: engine for key, engine in self._engines.items()
                if key in capabilities.engines}

    def describe(self, capabilities: Capabilities) -> list[dict]:
        """What the interface needs to draw the engine list.

        Unavailable engines are included, with the reason, rather than omitted.
        The binary path is reported too, so it is obvious which build is in use
        when two are installed.
        """
        rows = []
        supported = capabilities.supported_engines or frozenset(self._engines)
        for key, engine in self._engines.items():
            if key not in supported:
                continue
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
    if engine_id in ("vllm", "nemo") and capabilities.accelerator_kind != "cuda":
        return "Requires an NVIDIA GPU"
    if engine_id in ("vllm", "nemo", "mlxlm", "mlxvlm", "mlxwhisper", "onnx", "pyannote",
                     "paddleocr", "comfyui", "acestep", "qwentts", "kokoro"):
        return "Not installed"
    return "Binary not found"
