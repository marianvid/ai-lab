"""Coordinating the services.

Loading a model needs four things brought together: the configuration says
which instance, the catalog finds the model on disk, the registry supplies the
engine, and the runtime performs the move. None of those may import the
others, and the web layer must not make decisions — so the joining happens
here, one layer above the services and one below the routes.

Every method reads as the sentence a user would say: load this instance, swap
it to that model, download this one. That is the test for whether something
belongs in this file.
"""

from __future__ import annotations

import os
from dataclasses import asdict, replace
from pathlib import Path

# Where port suggestions start. Engines sit above the manager's own 8090 by
# convention, so the numbers read in the order things were added.
FIRST_PORT = 8080

# What an edit to an existing model is allowed to touch. Anything else is
# refused rather than ignored.
CHANGEABLE = frozenset({"params", "model_id", "port"})

from .application.instances import InstanceService
from .application.model_storage import ModelStorageService
from .application.downloads import ModelDownloadService
from .builds import Builds
from . import budget
from .capabilities import IMAGES, TOOLS
from .catalog import Catalog
from .changes import Reader, counted
from .config import INSTANCE_ID, ConfigStore, Instance, validate_distinct_roots
from .downloads import DownloadManager, HuggingFaceClient
from .engines.base import validate
from .hosts.base import Host
from .runtime import Operation, Runtime
from .types import ChangeEvent, Interests, LogEvent, Task
from .settings import Settings
from .storage import Storage


class Operations:
    def __init__(self, store: ConfigStore, catalog: Catalog, runtime: Runtime,
                 settings: Settings, downloads: DownloadManager,
                 huggingface: HuggingFaceClient, host: Host,
                 engines=None, builds: Builds | None = None,
                 installs: Installs | None = None,
                 storage: Storage | None = None,
                 bus=None, last_loaded=None) -> None:
        # `engines` is the engine registry. It is passed in rather than
        # imported for two reasons: the binary paths come from configuration,
        # and a test can supply engines that do not reach for the network when
        # asked whether they are ready.
        from .engines.registry import Registry
        self.engines = engines if engines is not None else Registry()
        self.builds = builds
        # Engines installed as packages rather than compiled. Optional:
        # a machine with none, and a test that does not care, pass nothing.
        self.installs = installs
        self.storage = storage
        self.bus = bus
        self.store = store
        self.catalog = catalog
        self.runtime = runtime
        self.settings = settings
        self.downloads = downloads
        self.huggingface = huggingface
        self.host = host
        # What was on the card when the manager last stopped, so it can be put
        # back. Optional: a test that does not care about it passes nothing,
        # and nothing is remembered.
        self.last_loaded = last_loaded
        self.image_jobs = None
        self.model_downloads = ModelDownloadService(
            store, downloads, huggingface, host, self.engines)
        self.model_storage = ModelStorageService(
            store, catalog, host, self._changed,
            self.model_downloads._writable_in_root)
        self.instance_service = InstanceService(
            store, catalog, runtime, host, self.engines, self._changed,
            self.models, bus=bus, last_loaded=last_loaded)

    def _changed(self, topic: str) -> None:
        """Tell whoever is watching that this kind of thing has moved.

        Only what changed, never the new value: the page fetches that itself,
        so this cannot drift into a second description of the state.
        """
        if self.bus is not None:
            self.bus.publish(ChangeEvent(topic=topic))

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

    def models(self, engine_id: str | None = None) -> list[dict]:
        """Every model on disk, optionally only those an engine can load."""
        config = self.store.load()
        models = self.catalog.scan(config.repositories)
        if engine_id:
            engine = self.engines.get(engine_id)
            formats = engine.formats()
            tasks = engine.tasks()
            models = [item for item in models
                      if item.format in formats and item.task in tasks
                      and (not hasattr(engine, "supports") or engine.supports(item))]
        rows = []
        for item in models:
            row = self._model(item)
            note = config.model_notes.get(item.id, config.model_notes.get(item.name, {}))
            if isinstance(note, str):
                note = {"short": note, "detail": note}
            if isinstance(note, dict):
                short = str(note.get("short", "")).strip()
                detail = str(note.get("detail", short)).strip()
                if short:
                    row["summary"] = short
                if detail:
                    row["description"] = detail
            try:
                row["storage_tier"] = config.repository(
                    item.id.split("/", 1)[0]).root_id
            except KeyError:
                # The prefix always comes from a configured repository's own
                # id (see `Catalog.scan`), so this cannot happen today — but
                # the listing itself must not fail if it ever does.
                row["storage_tier"] = ""
            rows.append(row)
        return rows

    def configured(self):
        return self.instance_service.configured()

    def instances(self):
        return self.instance_service.instances()

    def instance(self, instance_id: str):
        return self.instance_service.instance(instance_id)

    def model_for(self, instance_id: str):
        return self.instance_service.model_for(instance_id)

    def load(self, instance_id: str, settings: dict | None = None):
        return self.instance_service.load(instance_id, settings)

    def effective_params(self, instance_id: str, settings: dict):
        return self.instance_service.effective_params(instance_id, settings)

    def unload(self, instance_id: str):
        return self.instance_service.unload(instance_id)

    def restore_last(self):
        return self.instance_service.restore_last()

    def logs(self, instance_id: str, lines: int = 200):
        return self.instance_service.logs(instance_id, lines)

    def suggest_port(self):
        return self.instance_service.suggest_port()

    def new_instance_form(self):
        return self.instance_service.new_instance_form()

    def create_instance(self, payload: dict):
        return self.instance_service.create_instance(payload)

    def update_instance(self, instance_id: str, changes: dict):
        return self.instance_service.update_instance(instance_id, changes)

    def apply_and_reload(self, instance_id: str, changes: dict):
        return self.instance_service.apply_and_reload(instance_id, changes)

    def delete_instance(self, instance_id: str):
        return self.instance_service.delete_instance(instance_id)

    @staticmethod
    def _task(config, model_id: str) -> str:
        return InstanceService._task(config, model_id)

    @staticmethod
    def _effective(engine, stored: dict,
                   task: Task = Task.TEXT_GENERATION) -> dict:
        return InstanceService._effective(engine, stored, task)

    # -- keeping the engines up to date ------------------------------------

    # -- the front door's own settings -------------------------------------

    GATEWAY_SETTINGS = {
        "first_byte_s": (1.0, 3600.0),
        "between_bytes_s": (1.0, 3600.0),
        "max_waiting": (1, 10000),
    }

    def gateway_settings(self) -> dict:
        return dict(self.store.load().gateway)

    def update_gateway(self, changes: dict) -> dict:
        """Change how long the front door waits and how many it holds.

        Checked here rather than in the gateway, for the same reason every
        other setting is checked outside the thing it configures: a number that
        cannot work should be refused while it is being typed, not discovered
        when a request hangs.
        """
        unknown = set(changes) - set(self.GATEWAY_SETTINGS)
        if unknown:
            raise ValueError(f"Unknown settings: {', '.join(sorted(unknown))}")
        cleaned = {}
        for key, value in changes.items():
            low, high = self.GATEWAY_SETTINGS[key]
            try:
                number = int(value) if isinstance(low, int) else float(value)
            except (TypeError, ValueError):
                raise ValueError(f"{key} must be a number") from None
            if not low <= number <= high:
                raise ValueError(f"{key} must be between {low} and {high}")
            cleaned[key] = number
        with self.store.mutate() as config:
            config.gateway = {**config.gateway, **cleaned}
        self._changed("settings")
        return self.gateway_settings()

    # -- how much of this machine models may use ---------------------------

    def reserve_mb(self) -> float:
        """How much of this machine is held back for the machine itself."""
        return self.settings.reserve_mb(self.store.load())

    def memory_budget(self) -> dict:
        """What is available for models right now, pool by pool."""
        return budget.of(self.host, self.reserve_mb()).json()

    def update_memory(self, changes: dict) -> dict:
        """Change how much of the machine is held back for the machine.

        The upper limit is deliberately generous rather than tied to how much
        memory this machine has: a container can be given more, and a setting
        that refused the number somebody wants because of what the machine used
        to have would be worse than one that lets them hold back too much and
        see it on the page.
        """
        unknown = set(changes) - {"reserve_mb"}
        if unknown:
            raise ValueError(f"Unknown settings: {', '.join(sorted(unknown))}")
        cleaned = {}
        if "reserve_mb" in changes:
            try:
                number = float(changes["reserve_mb"])
            except (TypeError, ValueError):
                raise ValueError("reserve_mb must be a number") from None
            if not 0 <= number <= 1024 * 1024:
                raise ValueError("reserve_mb must be between 0 and 1048576")
            cleaned["reserve_mb"] = number
        with self.store.mutate() as config:
            config.memory = {**config.memory, **cleaned}
        self._changed("settings")
        return self.memory_budget()

    def build_status(self, with_sizes: bool = True) -> list[dict]:
        return self.builds.all(with_sizes=with_sizes) if self.builds else []

    def check_for_update(self, engine_id: str) -> dict:
        return self.builds.get(engine_id).check()

    # -- reading an update before taking it --------------------------------

    def what_would_change(self, engine_id: str) -> dict:
        """Everything that can be found out about updating this engine.

        An update should be a decision rather than a hope, so this is what the
        page shows before offering the button: what is installed, what it would
        become, which changes matter *here*, what the people upstream wrote
        about it, and — for an engine installed as packages — exactly which
        packages would be replaced and which of them would go backwards.

        Nothing is changed by asking. Every source is read separately, so one
        being unreachable costs only that section.
        """
        settings = self.store.load().engines.get(engine_id) or {}
        engine = self.engines.get(engine_id)
        reader = Reader(engine_id, settings.get("source"),
                        binary=settings.get("binary", ""))
        installed, latest = "", ""
        try:
            build = self.builds.get(engine_id)
            status = build.status()
            installed = status.get("installed") or ""
            latest = status.get("latest") or ""
        except KeyError:
            pass                       # not built from source; notes still apply
        if not installed or not latest:
            try:
                managed = self._installs().get(engine_id).status()
                installed = installed or managed.get("installed", "")
                latest = latest or managed.get("latest", "")
            except KeyError:
                pass
        # For an engine installed as packages there is no build to ask, so the
        # versions come from the environment and from what the package manager
        # says it would do. Asked before the notes are fetched, because the
        # notes are chosen by which versions lie between the two.
        moves, trouble = reader.moves()
        if not installed or not latest:
            from_packages = reader.versions(moves)
            installed = installed or from_packages[0]
            latest = latest or from_packages[1]
        found = reader.read(installed, latest, self.interests())
        return {
            "engine": engine_id,
            "engine_name": getattr(engine, "name", engine_id),
            "installed": found.installed,
            "latest": found.latest,
            "yours": [asdict(item) for item in found.yours],
            "others": [asdict(item) for item in found.others],
            "by_area": counted(found.yours),
            "other_areas": counted(found.others),
            "notes": found.notes,
            "packages": [move.json() for move in moves],
            "unreadable": " ".join(part for part in
                                   (found.unreadable, trouble) if part),
        }

    def interests(self) -> Interests:
        """What this machine uses, so an update can be read against it.

        Worked out, never written down. The card decides which hardware
        changes are worth reading — the Mac wants Metal and the container wants
        CUDA, and neither is told which it is. The configured entries decide
        the rest: somebody running only text models does not need a hundred
        lines about the vision code.
        """
        config = self.store.load()
        models = {item.id: item for item in self.catalog.scan(config.repositories)}
        formats, pictures, tools = set(), False, False
        for entry in config.instances:
            model = models.get(entry.model_id)
            if model is None:
                continue
            formats.add(model.format.value)
            able = model.capabilities
            # An entry that switches pictures off is not using them, however
            # capable its weights are.
            if IMAGES in able and not entry.params.get("language_model_only"):
                pictures = True
            if TOOLS in able:
                tools = True
        return Interests(accelerator_kind=self.host.capabilities().accelerator_kind,
                         formats=frozenset(formats), pictures=pictures,
                         tools=tools)

    # -- engines that arrive as packages -----------------------------------

    def install_status(self, engine_id: str) -> dict:
        return self._installs().get(engine_id).status()

    def installs_available(self) -> list[dict]:
        return self.installs.all() if self.installs else []

    def _installs(self) -> Installs:
        if self.installs is None:
            raise KeyError("No engine on this machine is installed as packages")
        return self.installs

    def install_engine(self, engine_id: str, version: str = "") -> dict:
        """Install a new version beside the one in use.

        Refused while anything is running, for the same reason recompiling is:
        the engine ends up being launched from somewhere else, and a model
        already on the card would keep running the old one while the page said
        otherwise. Better to be plain about it than to be subtly wrong.

        The download itself is safe at any time — nothing existing is written
        to — but the swap at the end is not worth splitting into a second
        button nobody would remember to press.
        """
        self._nothing_running("The engine is about to be launched from "
                              "somewhere else.")
        return self._installs().get(engine_id).install(version)

    def activate_install(self, engine_id: str, name: str) -> dict:
        """Go back to, or forward to, an installed version."""
        self._nothing_running("The engine is about to be launched from "
                              "somewhere else.")
        return self._installs().get(engine_id).activate(name)

    def remove_install(self, engine_id: str, name: str) -> dict:
        """Delete an installed version that is not in use.

        Never automatic. The previous version is the way back from an update
        that turned out badly, and deciding it is no longer needed is a
        judgement about whether the new one has proved itself — which is not a
        judgement a timer can make.
        """
        return self._installs().get(engine_id).remove(name)

    def update_install_component(self, engine_id: str, name: str) -> dict:
        self._nothing_running("The engine and its extensions are being replaced.")
        install = self._installs().get(engine_id)
        if not hasattr(install, "update_component"):
            raise ValueError(f"{engine_id} has no separately managed components")
        return install.update_component(name)

    def activate_build(self, engine_id: str, name: str) -> dict:
        """Select a compiled source version without rebuilding it."""
        self._nothing_running("The engine is about to be launched from "
                              "somewhere else.")
        return self.builds.get(engine_id).activate(name)

    def remove_build(self, engine_id: str, name: str) -> dict:
        """Delete a compiled source version that is not in use."""
        return self.builds.get(engine_id).remove(name)

    # -- disk space that is not model storage ------------------------------

    def storage_view(self) -> dict:
        return self.storage.view() if self.storage else {
            "items": [], "recoverable_bytes": 0}

    def clear_storage(self, item_id: str) -> dict:
        if self.storage is None:
            raise KeyError("No reclaimable storage is configured")
        return self.storage.clear(item_id)

    def _nothing_running(self, why: str) -> None:
        running = [item["id"] for item in self.instances() if item["running"]]
        if running:
            raise ValueError("Unload the running instances first: "
                             + ", ".join(running) + ". " + why)

    def update_engine(self, engine_id: str) -> dict:
        """Pull and recompile an engine from source.

        Refused while anything is running. On Linux the linker cannot write
        over a binary that is executing, so the build would fail partway with
        a confusing message about a busy file. Better to say plainly that the
        models need unloading first.
        """
        self._nothing_running("The engine binary cannot be replaced while it "
                              "is executing.")
        return self.builds.get(engine_id).update()

    # -- choosing where models live ----------------------------------------

    def browse(self, path: str | None = None, programs: bool = False) -> dict:
        """List what is inside one directory, for picking a path.

        A web page cannot open a file dialog on the machine the server runs on,
        so the server has to offer the listing itself.

        Folders always. Files only when `programs` is asked for, and then only
        ones that can be launched — not shared libraries, which carry the
        execute bit and cannot. Never file *contents*: this says what is there,
        and the less it can reach the better.
        """
        start = Path(path).expanduser() if path else self._default_browse_root()
        start = start.resolve()
        if not start.is_dir():
            raise ValueError(f"{start} is not a directory")

        entries = []
        try:
            for item in sorted(start.iterdir()):
                if item.name.startswith("."):
                    continue
                if item.is_dir():
                    entries.append({
                        "name": item.name, "path": str(item), "kind": "folder",
                        "writable": os.access(item, os.W_OK | os.X_OK),
                    })
                elif (programs and item.is_file() and os.access(item, os.X_OK)
                      and not _is_library(item.name)):
                    entries.append({
                        "name": item.name, "path": str(item), "kind": "program",
                        "writable": False,
                    })
        except PermissionError:
            raise ValueError(f"No permission to read {start}") from None

        return {
            "path": str(start),
            "parent": str(start.parent) if start.parent != start else None,
            "writable": os.access(start, os.W_OK | os.X_OK),
            "entries": entries,
        }

    def _default_browse_root(self) -> Path:
        """Start where the models already are, not at the root of the disk."""
        repositories = self.store.load().repositories
        for item in repositories:
            path = Path(item.path)
            if path.is_dir():
                return path.parent
        return Path.home()

    def update_engine_binary(self, engine_id: str, path: str) -> dict:
        """Point an engine at a different program.

        Checked before it is saved: a path that is not there means the engine
        shows as not installed on every screen afterwards, and the moment to
        say so is while the person is still looking at what they picked.

        Takes effect the next time a model starts. Nothing already running is
        touched — its process was launched from wherever it was launched from,
        and stopping somebody's model because a path was corrected would be a
        surprise nobody asked for.
        """
        if engine_id not in self.store.load().engines:
            raise KeyError(f"Unknown engine: {engine_id}")
        program = Path(str(path)).expanduser()
        if not program.exists():
            raise ValueError(f"{program} is not there")
        if program.is_dir():
            raise ValueError(f"{program} is a directory, not a program")
        if not os.access(program, os.X_OK):
            raise ValueError(f"{program} cannot be run")
        with self.store.mutate() as config:
            config.engines[engine_id] = {**config.engines[engine_id],
                                         "binary": str(program)}
        self._changed("settings")
        return {"engine": engine_id, "binary": str(program)}

    def update_models_root(self, path: str) -> dict:
        """Point every repository somewhere else, in one move.

        There is one root and each format is a folder in it. Setting them
        separately let GGUF end up on one disk and NVFP4 on another, which is
        a state nothing else in this application expects and which nobody
        chooses on purpose — `MODEL_STORAGE.md` has described the format-first
        tree as the layout all along.

        Checked before it is saved. A root that does not exist shows up as
        broken on every screen afterwards, and the moment to say so is while
        the person is still looking at the field they typed into.
        """
        root = Path(str(path)).expanduser()
        if not root.is_dir():
            raise ValueError(f"{root} is not a directory")
        with self.store.mutate() as config:
            config.models_root = str(root.resolve())
            config.model_root("core").path = config.models_root
            validate_distinct_roots(config.model_roots)
        self._changed("models")
        return {"models_root": self.store.load().models_root}

    def update_model_root(self, root_id: str, changes: dict) -> dict:
        allowed = {"path", "enabled", "writable", "download_default"}
        unknown = set(changes) - allowed
        if unknown:
            raise ValueError(
                f"Unknown model-root settings: {', '.join(sorted(unknown))}")
        with self.store.mutate() as config:
            model_root = config.model_root(root_id)
            if "path" in changes:
                path = Path(str(changes["path"])).expanduser()
                if not path.is_dir():
                    raise ValueError(f"{path} is not a directory")
                model_root.path = str(path.resolve())
                if root_id == "core":
                    config.models_root = model_root.path
            if "enabled" in changes:
                model_root.enabled = bool(changes["enabled"])
            if "writable" in changes:
                model_root.writable = bool(changes["writable"])
            if changes.get("download_default"):
                if not model_root.enabled:
                    raise ValueError(
                        "A disabled model root cannot receive downloads")
                config.download_root = root_id
            elif changes.get("download_default") is False \
                    and config.download_root == root_id:
                config.download_root = "core"
            validate_distinct_roots(config.model_roots)
        self._changed("settings")
        self._changed("models")
        return self.settings_view()

    def update_repository(self, repository_id: str, changes: dict) -> dict:
        """Rename a repository, or say whether it may be written to.

        Its path is not among these: it comes from the models root and this
        repository's id. See `update_models_root`.
        """
        allowed = {"name", "writable"}
        unknown = set(changes) - allowed
        if unknown:
            raise ValueError(
                f"Cannot change: {', '.join(sorted(unknown))}"
                + (". Every repository sits under the models root; set that "
                   "instead." if "path" in unknown else ""))

        with self.store.mutate() as config:
            repository = config.repository(repository_id)
            for key, value in changes.items():
                setattr(repository, key, value)
        self._changed("models")
        return asdict(self.store.load().repository(repository_id))

    def create_directory(self, path: str) -> dict:
        """Make a directory, so a repository can point at somewhere new.

        Offered because the alternative is telling someone to go and use a
        terminal in the middle of filling in a form.
        """
        target = Path(str(path)).expanduser()
        if target.exists():
            if target.is_dir():
                return {"path": str(target.resolve()), "created": False}
            raise ValueError(f"{target} exists and is not a directory")
        try:
            target.mkdir(parents=True)
        except OSError as error:
            raise ValueError(f"Could not create {target}: {error}") from None
        return {"path": str(target.resolve()), "created": True}

    # -- the library -------------------------------------------------------

    def supported_formats(self) -> list[str]:
        return self.model_downloads.supported_formats()

    def delete_model(self, model_id: str) -> dict:
        return self.model_storage.delete_model(model_id)

    def move_model(self, model_id: str, target_root_id: str) -> dict:
        return self.model_storage.move_model(model_id, target_root_id)

    def move_jobs(self) -> list[dict]:
        return self.model_storage.move_jobs()

    def recover_moves(self) -> list[dict]:
        return self.model_storage.recover_moves()

    def cancel_move(self, job_id: str) -> dict:
        return self.model_storage.cancel_move(job_id)

    # -- downloads ---------------------------------------------------------

    def search(self, query: str) -> dict:
        return self.model_downloads.search(query)

    def remote_sets(self, repo: str) -> list[dict]:
        return self.model_downloads.remote_sets(repo)

    def download(self, repo: str, name: str, repository_id: str | None = None,
                 storage_tier: str | None = None) -> dict:
        return self.model_downloads.download(repo, name, repository_id, storage_tier)

    def transfers(self) -> list[dict]:
        return self.model_downloads.transfers()

    def cancel_download(self, transfer_id: str) -> None:
        self.model_downloads.cancel_download(transfer_id)

    # -- internals ---------------------------------------------------------

    def _resolve(self, instance_id: str):
        return self.instance_service._resolve(instance_id)

    @staticmethod
    def _model(model) -> dict:
        return {"id": model.id, "name": model.name, "format": model.format.value,
                "task": model.task.value,
                "entrypoint": model.entrypoint, "size_bytes": model.size_bytes,
                "file_count": len(model.files), "complete": model.complete,
                "missing": list(model.missing),
                "capabilities": sorted(model.capabilities)}



# A shared library carries the execute bit and cannot be launched. Measured in
# llama.cpp's build directory on the container: 125 executable files, 33 of
# them `.so` companions to the launchers beside them.
LIBRARIES = (".so", ".dylib", ".dll")


def _is_library(name: str) -> bool:
    lowered = name.lower()
    return any(lowered.endswith(end) or f"{end}." in lowered for end in LIBRARIES)
