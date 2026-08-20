"""Building the objects and connecting them. Nothing else.

This is the only place that knows which concrete host is in use and where the
configuration lives. Everything else receives what it needs as an argument,
which is what lets tests substitute a fake host and a fake engine.

If logic starts appearing in this file, it belongs in `operations.py`.
"""

from __future__ import annotations

from pathlib import Path

from .builds import Builds
from .catalog import Catalog
from .config import ConfigStore
from .downloads import DownloadManager, HuggingFaceClient
from .engines.registry import Registry
from .events import EventBus
from .gateway import Gateway
from .hosts import current_host
from .operations import Operations
from .runtime import Runtime
from .settings import Settings


def build(config_path: Path) -> tuple[Operations, EventBus, ConfigStore, Gateway]:
    store = ConfigStore(config_path)
    bus = EventBus()
    # Engines are built from configuration, so a machine holding two llama.cpp
    # builds uses the one named in config.json rather than whichever PATH
    # happens to find first. The host needs the same section: vLLM lives in a
    # virtual environment, so whether it is installed cannot be read from PATH.
    engine_settings = store.load().engines
    host = current_host(engine_settings)
    engines = Registry(engine_settings)
    builds = Builds(engine_settings, bus)
    # Keeps the version figures current without anyone pressing anything.
    builds.watch()
    operations = Operations(
        store=store,
        catalog=Catalog(),
        runtime=Runtime(host, bus),
        settings=Settings(store, host, engines),
        engines=engines,
        downloads=DownloadManager(bus=bus),
        huggingface=HuggingFaceClient(),
        host=host,
        builds=builds,
        bus=bus,
    )
    # Routes a request by model name and loads that model if it is not running,
    # so an agent workflow can name several models and reach one card.
    return operations, bus, store, Gateway(operations)
