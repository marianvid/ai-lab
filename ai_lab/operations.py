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
from dataclasses import asdict
from pathlib import Path

# Where port suggestions start. Engines sit above the manager's own 8090 by
# convention, so the numbers read in the order things were added.
FIRST_PORT = 8080

from .builds import Builds
from .catalog import Catalog
from .config import ConfigStore, Instance
from .downloads import DownloadManager, HuggingFaceClient
from .engines.base import validate
from .hosts.base import Host
from .runtime import Operation, Runtime
from .types import ChangeEvent
from .settings import Settings


class Operations:
    def __init__(self, store: ConfigStore, catalog: Catalog, runtime: Runtime,
                 settings: Settings, downloads: DownloadManager,
                 huggingface: HuggingFaceClient, host: Host,
                 engines=None, builds: Builds | None = None,
                 bus=None) -> None:
        # `engines` is the engine registry. It is passed in rather than
        # imported for two reasons: the binary paths come from configuration,
        # and a test can supply engines that do not reach for the network when
        # asked whether they are ready.
        from .engines.registry import Registry
        self.engines = engines if engines is not None else Registry()
        self.builds = builds
        self.bus = bus
        self.store = store
        self.catalog = catalog
        self.runtime = runtime
        self.settings = settings
        self.downloads = downloads
        self.huggingface = huggingface
        self.host = host

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
        sources = {item["engine"]: item for item in self.build_status()}
        for engine in view["engines"]:
            engine["source"] = sources.get(engine["id"])
        return view

    def models(self, engine_id: str | None = None) -> list[dict]:
        """Every model on disk, optionally only those an engine can load."""
        config = self.store.load()
        models = self.catalog.scan(config.repositories)
        if engine_id:
            formats = self.engines.get(engine_id).formats()
            models = [item for item in models if item.format in formats]
        return [self._model(item) for item in models]

    def instances(self) -> list[dict]:
        """Every configured model, with the settings that will actually apply.

        Stored settings are filled in with the engine's defaults before being
        reported. An entry written before a setting existed has no value for
        it, and showing that as blank would be a lie: the engine will use its
        default, and that is what the interface should say.
        """
        config = self.store.load()
        rows = []
        for item in config.instances:
            engine = self.engines.get(item.engine)
            row = self.runtime.status(item, engine)
            row["params"] = self._effective(engine, item.params)
            rows.append(row)
        return rows

    @staticmethod
    def _effective(engine, stored: dict) -> dict:
        try:
            return validate(engine.params(), {key: value for key, value in stored.items()
                                              if key in {spec.key for spec in engine.params()}})
        except ValueError:
            return dict(stored)

    # -- moving models on and off the accelerator --------------------------

    def load(self, instance_id: str) -> Operation:
        """Start this model, replacing whatever this entry was running.

        One entry is one model, so there is no separate "swap": reloading with
        different settings and starting for the first time are the same act
        from the outside.
        """
        instance, model = self._resolve(instance_id)
        engine = self.engines.get(instance.engine)
        if self.host.status(instance_id).running:
            return self.runtime.swap(instance, model, engine)
        return self.runtime.load(instance, model, engine)

    def unload(self, instance_id: str) -> Operation:
        return self.runtime.unload(instance_id)

    # -- configuring instances ---------------------------------------------

    def suggest_port(self) -> int:
        """The first free port at or above 8080.

        Offered when adding a model so there is one less thing to think about,
        and still editable, because a port sometimes has to match what a client
        already expects.
        """
        taken = {item.port for item in self.store.load().instances}
        port = FIRST_PORT
        while port in taken:
            port += 1
        return port

    def new_instance_form(self) -> dict:
        """Everything the Add form needs, in one call.

        Which models are on disk, which engines can read them, what each engine
        can be tuned with, and a free port.
        """
        capabilities = self.host.capabilities()
        return {
            "port": self.suggest_port(),
            "engines": self.engines.describe(capabilities),
            "models": self.models(),
        }

    def create_instance(self, payload: dict) -> dict:
        engine = self.engines.get(payload["engine"])
        params = validate(engine.params(), payload.get("params", {}))
        instance = Instance(
            id=payload["id"], name=payload.get("name") or payload["id"],
            engine=payload["engine"], model_id=payload["model_id"],
            port=int(payload["port"]), params=params,
        )
        with self.store.mutate() as config:
            if any(item.id == instance.id for item in config.instances):
                raise ValueError(f"Instance {instance.id} already exists")
            if any(item.port == instance.port for item in config.instances):
                raise ValueError(f"Port {instance.port} is already in use")
            config.instances.append(instance)
        self._changed("instances")
        return asdict(instance)

    def update_instance(self, instance_id: str, changes: dict) -> dict:
        """Change the settings of an instance. Does not restart it.

        Saving and applying are separate acts, because applying means
        restarting, and restarting unloads a model somebody may be using.
        """
        with self.store.mutate() as config:
            instance = config.instance(instance_id)
            engine = self.engines.get(instance.engine)
            if "params" in changes:
                instance.params = validate(engine.params(), changes["params"])
            if "name" in changes:
                instance.name = str(changes["name"])
            if "model_id" in changes:
                instance.model_id = str(changes["model_id"])
        self._changed("instances")
        running = self.host.status(instance_id).running
        return {"id": instance_id, "applied": not running,
                "note": "" if not running
                        else "Reload to apply the new settings"}

    def apply_and_reload(self, instance_id: str, changes: dict) -> dict:
        """Save the settings and restart the model with them.

        The two halves are one action here because that is what the user
        means: the settings decide how much is reserved on the accelerator, so
        they only take effect when the model starts again.
        """
        self.update_instance(instance_id, changes)
        operation = self.load(instance_id)
        return {"id": instance_id, "applied": operation.ok,
                "operation": operation.json()}

    def delete_instance(self, instance_id: str) -> None:
        if self.host.status(instance_id).running:
            raise ValueError("Stop the instance before deleting it")
        with self.store.mutate() as config:
            config.instance(instance_id)                # raises if unknown
            config.instances = [item for item in config.instances
                                if item.id != instance_id]
        self._changed("instances")

    # -- keeping the engines up to date ------------------------------------

    def build_status(self) -> list[dict]:
        return self.builds.all() if self.builds else []

    def check_for_update(self, engine_id: str) -> dict:
        return self.builds.get(engine_id).check()

    def update_engine(self, engine_id: str) -> dict:
        """Pull and recompile an engine from source.

        Refused while anything is running. On Linux the linker cannot write
        over a binary that is executing, so the build would fail partway with
        a confusing message about a busy file. Better to say plainly that the
        models need unloading first.
        """
        running = [item["id"] for item in self.instances() if item["running"]]
        if running:
            raise ValueError(
                "Unload the running instances first: " + ", ".join(running)
                + ". The engine binary cannot be replaced while it is executing.")
        return self.builds.get(engine_id).update()

    # -- choosing where models live ----------------------------------------

    def browse(self, path: str | None = None) -> dict:
        """List the directories inside one directory, for picking a path.

        A web page cannot open a file dialog on the machine the server runs on,
        so the server has to offer the listing itself. Only directory names are
        returned, never file contents: this is for choosing a folder, and the
        less it can reach the better.
        """
        start = Path(path).expanduser() if path else self._default_browse_root()
        start = start.resolve()
        if not start.is_dir():
            raise ValueError(f"{start} is not a directory")

        entries = []
        try:
            for item in sorted(start.iterdir()):
                if item.name.startswith(".") or not item.is_dir():
                    continue
                entries.append({
                    "name": item.name,
                    "path": str(item),
                    "writable": os.access(item, os.W_OK | os.X_OK),
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

    def update_repository(self, repository_id: str, changes: dict) -> dict:
        """Point a repository somewhere else, or rename it.

        The path is checked before it is saved. A repository that does not
        exist shows up as broken on every screen afterwards, and the moment to
        say so is while the person is still looking at the field they typed
        into.
        """
        allowed = {"name", "path", "writable"}
        unknown = set(changes) - allowed
        if unknown:
            raise ValueError(f"Cannot change: {', '.join(sorted(unknown))}")

        if "path" in changes:
            path = Path(str(changes["path"])).expanduser()
            if not path.is_dir():
                raise ValueError(f"{path} is not a directory")
            changes = {**changes, "path": str(path.resolve())}

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
        """Formats something on this machine can actually run.

        Used to filter what is offered for download: there is no point pulling
        30 GB of safetensors onto a machine with no engine that reads them.
        """
        capabilities = self.host.capabilities()
        formats: set[str] = set()
        for engine in self.engines.available(capabilities).values():
            formats.update(item.value for item in engine.formats())
        return sorted(formats)

    def delete_model(self, model_id: str) -> dict:
        """Remove a model's files from disk.

        Refused while any configured entry points at it — deleting the files
        under a running model would leave a process serving weights that no
        longer exist, and under a stopped one an entry that can never start.
        Removing the entry first is one click, and it makes the order of events
        the user's decision rather than a surprise.

        Every path is checked against the configured repositories before
        anything is unlinked. The model id arrives over HTTP, and this is the
        one operation in the application that destroys data.
        """
        config = self.store.load()
        model = self.catalog.find(config.repositories, model_id)
        users = [item.name for item in config.instances if item.model_id == model_id]
        if users:
            raise ValueError(
                "Remove the entry from the Models tab first: "
                + ", ".join(users) + " still points at this model.")

        roots = [Path(item.path).resolve() for item in config.repositories]
        paths = [Path(item.path).resolve() for item in model.files]
        for path in paths:
            if not any(path.is_relative_to(root) for root in roots):
                raise ValueError(f"Refusing to delete outside the repositories: {path}")

        freed = sum(item.size_bytes for item in model.files)
        for path in paths:
            path.unlink(missing_ok=True)
        self._prune_empty(paths, roots)
        self._changed("models")
        return {"deleted": model.name, "files": len(paths), "freed_bytes": freed}

    @staticmethod
    def _prune_empty(paths: list[Path], roots: list[Path]) -> None:
        """Take away the directory too, if the model was the only thing in it.

        A model usually lives in its own directory, and leaving empty ones
        behind makes the library look like it still holds something.
        """
        for directory in {path.parent for path in paths}:
            if directory in roots:
                continue
            if any(directory.iterdir()):
                continue
            if any(directory.is_relative_to(root) for root in roots):
                directory.rmdir()

    # -- downloads ---------------------------------------------------------

    def search(self, query: str, only_supported: bool = True) -> list[dict]:
        results = self.huggingface.search(query)
        if not only_supported:
            return results
        supported = set(self.supported_formats())
        return [item for item in results if supported.intersection(item["formats"])]

    def remote_sets(self, repo: str, only_supported: bool = True) -> list[dict]:
        """What a repository holds, by default only what this machine can run."""
        supported = set(self.supported_formats())
        return [item.json() for item in self.huggingface.sets(repo)
                if not only_supported or item.format in supported]

    def download(self, repo: str, name: str, repository_id: str | None = None) -> dict:
        """Queue a complete model, into the repository that holds its format.

        The destination is worked out rather than asked for. A GGUF model
        belongs in the GGUF repository — the store is organised by format, and
        the listing already says which format this is, so making someone
        choose asks a question whose answer is already known. It can still be
        given explicitly when more than one repository holds a format.

        When it is given, it is checked first: a destination that cannot be
        written to should not cost a round trip to Hugging Face to discover.
        """
        config = self.store.load()

        if repository_id:
            destination = self._writable(config.repository(repository_id))
            remote = self._remote_set(repo, name)
        else:
            remote = self._remote_set(repo, name)
            destination = self._repository_for(config, remote.format)

        target = Path(destination.path) / Path(name).name
        return self.downloads.enqueue(remote, target).json()

    def _remote_set(self, repo: str, name: str):
        remote = next((item for item in self.huggingface.sets(repo)
                       if item.name == name), None)
        if remote is None:
            raise KeyError(f"{name} is not in {repo}")
        return remote

    def _repository_for(self, config, format_name: str):
        """The repository that holds this format and can be written to."""
        candidates = [item for item in config.repositories
                      if item.format == format_name]
        if not candidates:
            raise ValueError(
                f"No repository is configured for {format_name} models. "
                f"Add one in the configuration first.")
        errors = []
        for item in candidates:
            try:
                return self._writable(item)
            except ValueError as error:
                errors.append(str(error))
        raise ValueError(errors[0])

    @staticmethod
    def _writable(repository):
        """Return the repository, or explain why it cannot be written to.

        Writability is checked against the filesystem rather than the flag in
        the configuration: the flag records what was intended, and a download
        that starts and then dies on a permission error has wasted the wait.
        """
        path = Path(repository.path)
        if not repository.writable:
            raise ValueError(f"{repository.name} is marked read-only")
        if not path.is_dir():
            raise ValueError(f"{repository.name} does not exist at {path}")
        if not os.access(path, os.W_OK | os.X_OK):
            raise ValueError(
                f"{repository.name} is not writable by the manager. "
                f"Give it ownership of {path}.")
        return repository

    def transfers(self) -> list[dict]:
        return self.downloads.list()

    def cancel_download(self, transfer_id: str) -> None:
        self.downloads.cancel(transfer_id)

    # -- internals ---------------------------------------------------------

    def _resolve(self, instance_id: str):
        config = self.store.load()
        instance = config.instance(instance_id)
        model = self.catalog.find(config.repositories, instance.model_id)
        return instance, model

    @staticmethod
    def _model(model) -> dict:
        return {"id": model.id, "name": model.name, "format": model.format.value,
                "entrypoint": model.entrypoint, "size_bytes": model.size_bytes,
                "file_count": len(model.files), "complete": model.complete,
                "missing": list(model.missing)}
