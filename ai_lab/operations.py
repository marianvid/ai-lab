"""Coordinating the services.

Loading a model needs four things brought together: the configuration says
which instance, the catalog finds the model on disk, the registry supplies the
engine, and the runtime performs the move. None of those may import the
others, and the web layer must not make decisions — so the joining happens
here, one layer above the services and one below the routes.

Every method reads as the sentence a user would say: load this instance, swap
it to that model, download this one. That is the test for whether something
belongs in this file. The services themselves are built elsewhere
(`operations_services.build_services`, called by `wiring.build`) and handed in.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .application import paths
from .operations_services import OperationsParts, OperationsServices, build_services

if TYPE_CHECKING:
    from .images.jobs import ImageJobs
    from .media.jobs import MediaJobs
    from .types import Interests


class Operations:
    """What the routes can ask of this machine, each a sentence a person would say.

    A façade: every method hands the request to the service that owns it, or
    joins two of them (`settings_view`). It holds the parts it was built from
    so the routes and the gateway can reach the store, the host and the
    engines through one object.
    """

    def __init__(self, parts: OperationsParts, services: OperationsServices) -> None:
        self.parts = parts
        self.store = parts.store
        self.catalog = parts.catalog
        self.runtime = parts.runtime
        self.settings = parts.settings
        self.downloads = parts.downloads
        self.huggingface = parts.huggingface
        self.host = parts.host
        self.engines = parts.engines
        self.builds = parts.builds
        self.installs = parts.installs
        self.storage = parts.storage
        self.bus = parts.bus
        self.last_loaded = parts.last_loaded
        self.instance_service = services.instance_service
        self.model_storage = services.model_storage
        self._listing = services.listing
        self._downloads = services.model_downloads
        self._maintenance = services.engine_maintenance
        self._settings_changes = services.settings_changes
        # Set by `wiring.build` once the gateway exists (they need it).
        self.image_jobs: ImageJobs | None = None
        self.media_jobs: MediaJobs | None = None

    @classmethod
    def assemble(cls, parts: OperationsParts) -> Operations:
        """Built from its parts, with the standard services (see `build_services`)."""
        return cls(parts, build_services(parts))

    # -- reading -----------------------------------------------------------

    def settings_view(self) -> dict:
        """The settings screen, with each engine described in one piece.

        Which engines exist and which source checkouts exist are answered by
        two different modules, and joining them is this layer's job. Presenting
        them separately made the reader correlate two blocks that were about
        the same thing — the same engine, under two spellings, with two paths
        and two badges.
        """
        view = self.settings.view()
        sources = {item["engine"]: item for item in self.build_status(with_sizes=False)}
        for engine in view["engines"]:
            engine["source"] = sources.get(engine["id"])
        return view

    # -- the configured entries ------------------------------------------------

    def configured(self):
        """Every entry, without asking what any of them is doing."""
        return self.instance_service.configured()

    def instances(self):
        """Every configured model, with the settings that will actually apply."""
        return self.instance_service.instances()

    def instance(self, instance_id: str):
        """One configured entry, without asking what it is doing."""
        return self.instance_service.instance(instance_id)

    def model_for(self, instance_id: str):
        """The model an entry points at, for asking how big it is."""
        return self.instance_service.model_for(instance_id)

    def load(self, instance_id: str, settings: dict | None = None):
        """Start this model, replacing whatever this entry was running."""
        return self.instance_service.load(instance_id, settings)

    def effective_params(self, instance_id: str, settings: dict):
        """The entry's settings with these laid over them, checked."""
        return self.instance_service.effective_params(instance_id, settings)

    def unload(self, instance_id: str):
        """Stop this entry."""
        return self.instance_service.unload(instance_id)

    def restore_last(self):
        """Put back whatever was on the card before the manager stopped."""
        return self.instance_service.restore_last()

    def logs(self, instance_id: str, lines: int = 200):
        """What the engine has printed about itself."""
        return self.instance_service.logs(instance_id, lines)

    def suggest_port(self):
        """The first free port at or above 8080."""
        return self.instance_service.suggest_port()

    def new_instance_form(self):
        """Everything the Add form needs, in one call."""
        return self.instance_service.new_instance_form()

    def create_instance(self, payload: dict):
        """Add an entry. The id is given rather than worked out."""
        return self.instance_service.create_instance(payload)

    def update_instance(self, instance_id: str, changes: dict):
        """Change the settings of an instance. Does not restart it."""
        return self.instance_service.update_instance(instance_id, changes)

    def apply_and_reload(self, instance_id: str, changes: dict):
        """Save the settings and restart the model with them."""
        return self.instance_service.apply_and_reload(instance_id, changes)

    def delete_instance(self, instance_id: str):
        """Remove an entry that is not running."""
        return self.instance_service.delete_instance(instance_id)

    # -- this machine's settings -----------------------------------------------

    def gateway_settings(self) -> dict:
        """How long the front door waits and how many it holds, as saved."""
        return self._settings_changes.gateway_settings()

    def update_gateway(self, changes: dict) -> dict:
        """Change the front door's limits, checked before they are saved."""
        return self._settings_changes.update_gateway(changes)

    def reserve_mb(self) -> float:
        """How much of this machine is held back for the machine itself."""
        return self._settings_changes.reserve_mb()

    def memory_budget(self) -> dict:
        """What is available for models right now, pool by pool."""
        return self._settings_changes.memory_budget()

    def update_memory(self, changes: dict) -> dict:
        """Change how much of the machine is held back for the machine."""
        return self._settings_changes.update_memory(changes)

    def update_engine_binary(self, engine_id: str, path: str) -> dict:
        """Point an engine at a different program."""
        return self._settings_changes.update_engine_binary(engine_id, path)

    def update_models_root(self, path: str) -> dict:
        """Point every repository somewhere else, in one move."""
        return self._settings_changes.update_models_root(path)

    def update_repository(self, repository_id: str, changes: dict) -> dict:
        """Rename a repository, or say whether it may be written to."""
        return self._settings_changes.update_repository(repository_id, changes)

    # -- keeping the engines up to date ----------------------------------------

    def build_status(self, with_sizes: bool = True) -> list[dict]:
        """Every engine built from source: its checkout and compiled versions."""
        return self._maintenance.build_status(with_sizes)

    def check_for_update(self, engine_id: str) -> dict:
        """Whether a newer version of this engine is out."""
        return self._maintenance.check_for_update(engine_id)

    def what_would_change(self, engine_id: str) -> dict:
        """Everything that can be found out about updating this engine."""
        return self._maintenance.what_would_change(engine_id)

    def interests(self) -> Interests:
        """What this machine uses, so an update can be read against it."""
        return self._maintenance.interests()

    def install_status(self, engine_id: str) -> dict:
        """This packaged engine's installed versions and what is under way."""
        return self._maintenance.install_status(engine_id)

    def installs_available(self) -> list[dict]:
        """Every packaged engine that can be installed here."""
        return self._maintenance.installs_available()

    def install_engine(self, engine_id: str, version: str = "") -> dict:
        """Install a new version beside the one in use."""
        return self._maintenance.install_engine(engine_id, version)

    def activate_install(self, engine_id: str, name: str) -> dict:
        """Go back to, or forward to, an installed version."""
        return self._maintenance.activate_install(engine_id, name)

    def remove_install(self, engine_id: str, name: str) -> dict:
        """Delete an installed version that is not in use."""
        return self._maintenance.remove_install(engine_id, name)

    def update_install_component(self, engine_id: str, name: str) -> dict:
        """Update one component of an installed version."""
        return self._maintenance.update_install_component(engine_id, name)

    def activate_build(self, engine_id: str, name: str) -> dict:
        """Select a compiled source version without rebuilding it."""
        return self._maintenance.activate_build(engine_id, name)

    def remove_build(self, engine_id: str, name: str) -> dict:
        """Delete a compiled source version that is not in use."""
        return self._maintenance.remove_build(engine_id, name)

    def update_engine(self, engine_id: str) -> dict:
        """Pull and recompile an engine from source."""
        return self._maintenance.update_engine(engine_id)

    # -- the library -----------------------------------------------------------

    def models(self, engine_id: str | None = None) -> list[dict]:
        """Every model on disk, optionally only those an engine can load."""
        return self._listing.models(engine_id)

    def supported_formats(self) -> list[str]:
        """Formats something on this machine can actually run."""
        return self._downloads.supported_formats()

    def delete_model(self, model_id: str) -> dict:
        """Remove a model's files from disk."""
        return self.model_storage.delete_model(model_id)

    def move_model(self, model_id: str, target_root_id: str) -> dict:
        """Copy, verify and only then remove a model from its current tier."""
        return self.model_storage.move_model(model_id, target_root_id)

    def move_jobs(self) -> list[dict]:
        """Every move job on record, most recently updated first."""
        return self.model_storage.move_jobs()

    def recover_moves(self) -> list[dict]:
        """Sweep move jobs left mid-flight by a process that did not exit cleanly."""
        return self.model_storage.recover_moves()

    def cancel_move(self, job_id: str) -> dict:
        """Ask an in-progress move to stop at the next file or phase boundary."""
        return self.model_storage.cancel_move(job_id)

    # -- downloads -------------------------------------------------------------

    def search(self, query: str) -> dict:
        """Repositories holding something this machine can run."""
        return self._downloads.search(query)

    def remote_sets(self, repo: str) -> list[dict]:
        """What a repository holds that this machine can run."""
        return self._downloads.remote_sets(repo)

    def transfers(self) -> list[dict]:
        """The downloads under way or recently finished."""
        return self._downloads.transfers()

    def update_model_root(self, root_id: str, changes: dict) -> dict:
        """Change one model root; the settings screen as it is afterwards."""
        self._settings_changes.update_model_root(root_id, changes)
        return self.settings_view()

    def browse(self, path: str | None = None, programs: bool = False) -> dict:
        """List what is inside one directory, for picking a path (see `application.paths`)."""
        start = (Path(path) if path else
                 paths.browse_root([item.path for item in self.store.load().repositories]))
        return paths.browse(start, programs)

    def create_directory(self, path: str) -> dict:
        """Make a directory, so a repository can point at somewhere new."""
        return paths.create_directory(path)

    # -- disk space that is not model storage ------------------------------

    def storage_view(self) -> dict:
        """What can be reclaimed outside model storage, item by item."""
        return self.storage.view() if self.storage else {
            "items": [], "recoverable_bytes": 0}

    def clear_storage(self, item_id: str) -> dict:
        """Reclaim one item; KeyError when nothing reclaimable is configured."""
        if self.storage is None:
            raise KeyError("No reclaimable storage is configured")
        return self.storage.clear(item_id)

    # -- downloads, continued ------------------------------------------------

    def download(self, repo: str, name: str, repository_id: str | None = None,
                 storage_tier: str | None = None) -> dict:
        """Queue a complete model, into the repository that holds its format."""
        return self._downloads.download(repo, name, repository_id, storage_tier)

    def cancel_download(self, transfer_id: str) -> None:
        """Stop a download under way."""
        self._downloads.cancel_download(transfer_id)
