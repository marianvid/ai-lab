"""Building the objects and connecting them. Nothing else.

This is the only place that knows which concrete host is in use and where the
configuration lives. Everything else receives what it needs as an argument,
which is what lets tests substitute a fake host and a fake engine.

If logic starts appearing in this file, it belongs in `operations.py`.
"""

from __future__ import annotations

from pathlib import Path

from .builds import Builds
from .capabilities import Known
from .catalog import Catalog
from .config import ConfigStore
from .downloads import DownloadManager, HuggingFaceClient
from .engines.registry import Registry
from .events import EventBus
from .gateway import (BETWEEN_BYTES_S, FIRST_BYTE_S, MAX_WAITING,
                      Gateway)
from .hosts import current_host
from .installs import Installs
from .lastloaded import LastLoaded
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
    # Engines that arrive as packages rather than as source. A new
    # version is installed beside the one that works, never over it.
    installs = Installs(engine_settings, bus)
    # Keeps the version figures current without anyone pressing anything.
    builds.watch()
    # What each model's files say it can do, read once and kept beside the
    # rest of the state: a quarter of a second per GGUF model otherwise, on
    # every scan of the library.
    known = Known(host.state_dir())

    def learn_about_new_models(_destination) -> None:
        """Read a newly downloaded model's files now, not when a page asks.

        A full scan, because that is the only way in: everything already
        known is answered from memory, so what this actually costs is reading
        the one model that just arrived.
        """
        operations.models()

    operations = Operations(
        store=store,
        catalog=Catalog(known),
        runtime=Runtime(host, bus),
        settings=Settings(store, host, engines),
        engines=engines,
        downloads=DownloadManager(bus=bus, arrived=learn_about_new_models),
        huggingface=HuggingFaceClient(),
        host=host,
        builds=builds,
        installs=installs,
        bus=bus,
        # What was on the card when the manager last stopped. The host says
        # where a machine keeps such things.
        last_loaded=LastLoaded(host.state_dir()),
    )
    # Routes a request by model name and loads that model if it is not running,
    # so an agent workflow can name several models and reach one card. Its own
    # settings — how long to wait for an engine, how many requests to hold —
    # come from the configuration, because the right numbers differ between the
    # two machines this runs on.
    front_door = store.load().gateway
    return operations, bus, store, Gateway(
        operations,
        first_byte_s=float(front_door.get("first_byte_s", FIRST_BYTE_S)),
        between_bytes_s=float(front_door.get("between_bytes_s", BETWEEN_BYTES_S)),
        max_waiting=int(front_door.get("max_waiting", MAX_WAITING)))
