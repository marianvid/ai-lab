"""Inference engines.

One file per engine. Each declares what it can read, how it can be tuned, and
how it is launched — so adding an engine touches neither the configuration
schema nor the front end.

`Registry` is the entry point: it is built from configuration and handed to
whoever needs an engine, rather than being module state, because the path to
each engine's binary comes from config.json.
"""

from .base import Engine, LaunchPlan, ParamSpec, validate
from .registry import Registry, build

__all__ = ["Engine", "LaunchPlan", "ParamSpec", "validate", "Registry", "build"]
