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

# Keys an engine's section holds for the builds and installs that maintain
# the engine, not for the engine itself.
MAINTENANCE_KEYS = frozenset({"source", "update"})

# Modules of this package that hold no engine.
_NOT_ENGINES = frozenset({"base", "isolated", "probe", "registry", "text_launch"})


def engine_classes() -> dict[str, type]:
    """Every engine this package defines, by id, found by looking.

    An engine registers itself by being a module of this package, or of one of
    its sub-packages (one per kind of model), with a class that has an `id`, a
    `display_name` and a `plan` method — the shape of `base.Engine`. Adding
    one touches only its own file; nothing here lists them by name. Ordered by
    the module's own name, whatever sub-package it is in, so the list is the
    same each time. Two classes with the same `id` are refused, naming both:
    one would otherwise silently replace the other.
    """
    found: dict[str, type] = {}
    package = importlib.import_module(__package__ or "ai_lab.engines")
    for module_info in sorted(pkgutil.walk_packages(package.__path__, package.__name__ + "."),
                              key=lambda item: item.name.rsplit(".", 1)[-1]):
        leaf = module_info.name.rsplit(".", 1)[-1]
        if module_info.ispkg or leaf in _NOT_ENGINES or leaf.startswith("_"):
            continue
        module = importlib.import_module(module_info.name)
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if cls.__module__ == module.__name__ and _is_engine(cls):
                _add(found, cls)
    return found


def _add(found: dict[str, type], cls: type) -> None:
    """Register one engine class under its id; a second class with that id is an error."""
    engine_id = getattr(cls, "id", "")
    other = found.get(engine_id)
    if other is not None and other is not cls:
        raise ValueError(f"two engines share the id {engine_id!r}: "
                         f"{other.__module__}.{other.__name__} and {cls.__module__}.{cls.__name__}")
    found[engine_id] = cls


def _is_engine(cls: type) -> bool:
    """Whether a class has the shape of an engine (see `base.Engine`)."""
    return (isinstance(getattr(cls, "id", None), str)
            and isinstance(getattr(cls, "display_name", None), str)
            and callable(getattr(cls, "plan", None)))


def _construct(engine_id: str, cls: type, settings: dict) -> Engine:
    """One engine, given the settings its constructor takes.

    An engine's section of the configuration holds the constructor's settings
    plus `MAINTENANCE_KEYS`, which belong to the builds and installs that
    maintain it. Any other key is refused, naming it: a mistyped `binray`
    would otherwise be dropped and the engine started from whatever is on
    PATH — the silent pick this module exists to prevent. An engine with
    nothing configured falls back to its defaults (usually: look on PATH). An
    engine whose settings are grouped reads its section itself, with a
    `from_settings` class method.
    """
    unknown = sorted(set(settings) - accepted_settings(cls))
    if unknown:
        raise ValueError(f"engines.{engine_id}: unknown setting(s) {', '.join(unknown)}; "
                         f"it takes {', '.join(sorted(accepted_settings(cls) - MAINTENANCE_KEYS))}")
    make = getattr(cls, "from_settings", None)
    if make is not None:
        return make(settings)
    accepted = inspect.signature(cls).parameters
    return cls(**{key: value for key, value in settings.items() if key in accepted})


def accepted_settings(cls: type) -> set[str]:
    """The keys an engine's configuration section may hold."""
    names = set(inspect.signature(cls).parameters) - {"locations"}
    names |= set(getattr(cls, "location_settings", ()))
    return names | MAINTENANCE_KEYS


def build(settings: dict | None = None) -> dict[str, Engine]:
    """Create the engines, applying any per-engine configuration.

    `settings` is the `engines` section of config.json, for example
    `{"llamacpp": {"binary": "/opt/ai/llama.cpp/build/bin/llama-server"}}`.
    """
    settings = settings or {}
    return {engine_id: _construct(engine_id, cls, settings.get(engine_id) or {})
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
