"""What `Operations` is made of: the parts it is given and the services built from them.

`Operations` joins the services and adds nothing of its own (see
`operations.py`). It does not build them: `wiring.build` hands it its parts
and the services made from those parts (constructor injection), so a test can
build the same thing with fakes, and nothing in `operations.py` needs to know
how a service is put together.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .application.downloads import ModelDownloadService
from .application.engine_maintenance import EngineMaintenanceService, MaintenanceParts
from .application.instances import InstanceParts, InstanceService
from .application.model_listing import ModelListing
from .application.model_storage import ModelStorageService
from .application.settings_changes import SettingsChanges
from .types import ChangeEvent

if TYPE_CHECKING:
    from .builds import Builds
    from .catalog import Catalog
    from .config import ConfigStore
    from .downloads import DownloadManager, HuggingFaceClient
    from .hosts.base import Host
    from .installs import Installs
    from .runtime import Runtime
    from .settings import Settings
    from .storage import Storage


@dataclass(frozen=True)
class OperationsParts:
    """The objects `Operations` works with, built by `wiring.build` (or a test).

    `engines` is the engine registry, passed in rather than built here for two
    reasons: the binary paths come from configuration, and a test can supply
    engines that do not reach for the network when asked whether they are
    ready. `builds`, `installs`, `storage`, `bus` and `last_loaded` are
    optional: a machine with none, and a test that does not care, pass nothing.
    """

    store: ConfigStore
    catalog: Catalog
    runtime: Runtime
    settings: Settings
    downloads: DownloadManager
    huggingface: HuggingFaceClient
    host: Host
    engines: Any
    builds: Builds | None = None
    installs: Installs | None = None
    storage: Storage | None = None
    bus: Any = None
    # What was on the card when the manager last stopped, so it can be put back.
    last_loaded: Any = None


@dataclass(frozen=True)
class OperationsServices:
    """The services `Operations` delegates to, one per area of the application."""

    listing: ModelListing
    model_downloads: ModelDownloadService
    model_storage: ModelStorageService
    instance_service: InstanceService
    engine_maintenance: EngineMaintenanceService
    settings_changes: SettingsChanges


class ChangeNotifier:
    """Tells whoever is watching that one kind of thing has moved.

    Only what changed, never the new value: the page fetches that itself, so
    this cannot drift into a second description of the state.
    """

    def __init__(self, bus) -> None:
        self.bus = bus

    def __call__(self, topic: str) -> None:
        """Publish that `topic` changed; nothing when there is no bus."""
        if self.bus is not None:
            self.bus.publish(ChangeEvent(topic=topic))


def build_services(parts: OperationsParts) -> OperationsServices:
    """Every service, built from the parts and connected to each other."""
    changed = ChangeNotifier(parts.bus)
    listing = ModelListing(parts.store, parts.catalog, parts.engines)
    downloads = ModelDownloadService(parts.store, parts.downloads, parts.huggingface,
                                     parts.host, parts.engines)
    storage = ModelStorageService(parts.store, parts.catalog, parts.host, changed,
                                  downloads.writable_in_root)
    instances = InstanceService(InstanceParts(
        parts.store, parts.catalog, parts.runtime, parts.host, parts.engines, changed,
        listing.models, bus=parts.bus, last_loaded=parts.last_loaded))
    maintenance = EngineMaintenanceService(
        MaintenanceParts(parts.store, parts.catalog, parts.host, parts.engines,
                         parts.builds, parts.installs), instances.instances)
    return OperationsServices(
        listing=listing, model_downloads=downloads, model_storage=storage,
        instance_service=instances, engine_maintenance=maintenance,
        settings_changes=SettingsChanges(parts.store, parts.host, parts.settings, changed))
