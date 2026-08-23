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

from .builds import Builds
from . import budget
from .capabilities import IMAGES, TOOLS
from .catalog import Catalog
from .changes import Reader, counted
from .config import INSTANCE_ID, ConfigStore, Instance
from .downloads import DownloadManager, HuggingFaceClient
from .engines.base import validate
from .hosts.base import Host
from .runtime import Operation, Runtime
from .types import ChangeEvent, Interests, LogEvent
from .settings import Settings


class Operations:
    def __init__(self, store: ConfigStore, catalog: Catalog, runtime: Runtime,
                 settings: Settings, downloads: DownloadManager,
                 huggingface: HuggingFaceClient, host: Host,
                 engines=None, builds: Builds | None = None,
                 installs: Installs | None = None,
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

    def configured(self) -> list[dict]:
        """Every entry, without asking what any of them is doing.

        `instances` asks the supervisor about all of them and probes each one
        that is up: 73 ms on the container with eleven configured, nearly all
        of it the one command to systemd. Most questions are not about that —
        which entry answers to a name, which engine runs it, what settings it
        has — and those are the configuration, which costs 0.05 ms to read.
        """
        config = self.store.load()
        rows = []
        for item in config.instances:
            engine = self.engines.get(item.engine)
            rows.append({"id": item.id, "engine": item.engine,
                         "model_id": item.model_id, "port": item.port,
                         "params": self._effective(engine, item.params)})
        return rows

    def instances(self) -> list[dict]:
        """Every configured model, with the settings that will actually apply.

        Stored settings are filled in with the engine's defaults before being
        reported. An entry written before a setting existed has no value for
        it, and showing that as blank would be a lie: the engine will use its
        default, and that is what the interface should say.
        """
        config = self.store.load()
        # One question to the supervisor for the whole list rather than one per
        # entry. On systemd that was three commands each: eleven instances cost
        # 152 ms, which was the entire cost of this call, and the gateway asks
        # it twice on every request.
        processes = self.host.statuses([item.id for item in config.instances])
        rows = []
        for item in config.instances:
            engine = self.engines.get(item.engine)
            row = self.runtime.status(item, engine, processes.get(item.id))
            row["params"] = self._effective(engine, item.params)
            rows.append(row)
        return rows

    def instance(self, instance_id: str) -> dict:
        """One configured entry, without asking what it is doing. See `configured`."""
        item = self.store.load().instance(instance_id)     # raises if unknown
        engine = self.engines.get(item.engine)
        return {"id": item.id, "engine": item.engine,
                "model_id": item.model_id, "port": item.port,
                "params": self._effective(engine, item.params)}

    def model_for(self, instance_id: str):
        """The model an entry points at, for asking how big it is.

        Walks the model directories, so it is not on the path of a request that
        is going straight through — only of one that is about to cause a load,
        where a few milliseconds against a forty-second load is nothing.
        """
        config = self.store.load()
        instance = config.instance(instance_id)
        return self.catalog.find(config.repositories, instance.model_id)

    @staticmethod
    def _effective(engine, stored: dict) -> dict:
        try:
            return validate(engine.params(), {key: value for key, value in stored.items()
                                              if key in {spec.key for spec in engine.params()}})
        except ValueError:
            return dict(stored)

    # -- moving models on and off the accelerator --------------------------

    def load(self, instance_id: str, settings: dict | None = None) -> Operation:
        """Start this model, replacing whatever this entry was running.

        One entry is one model, so there is no separate "swap": reloading with
        different settings and starting for the first time are the same act
        from the outside.

        `settings` starts it with something other than what it is configured
        with, **without saving them**. A request can ask for a bigger context
        than the entry was set up for, and it would be wrong for that one
        request to quietly rewrite what somebody chose in the page. The running
        model differs from its configuration until it is unloaded, and says so.
        """
        instance, model = self._resolve(instance_id)
        engine = self.engines.get(instance.engine)
        if settings:
            instance = replace(instance,
                               params=self.effective_params(instance_id, settings))
        if self.host.status(instance_id).running:
            operation = self.runtime.swap(instance, model, engine)
        else:
            operation = self.runtime.load(instance, model, engine)
        if operation.ok and self.last_loaded:
            self.last_loaded.remember(instance_id, settings)
        return operation

    def effective_params(self, instance_id: str, settings: dict) -> dict:
        """The entry's settings with these laid over them, checked.

        Raises ValueError naming what is wrong, so a caller can refuse a
        request before anything is loaded rather than after. The engine's own
        rules do the checking, so a setting it does not have is refused here
        for the same reason it would be refused in the page.
        """
        # Not `_resolve`: that also finds the model on disk, which means
        # walking every model directory — 11 ms on the container, for an answer
        # made entirely of the configuration and the engine's own rules. This
        # is asked on every request through the front door.
        instance = self.store.load().instance(instance_id)
        engine = self.engines.get(instance.engine)
        return validate(engine.params(), {**instance.params, **settings})

    def unload(self, instance_id: str) -> Operation:
        operation = self.runtime.unload(instance_id)
        if operation.ok and self.last_loaded:
            # Named, so unloading a stray from beside the model that stays is
            # not read as the card having been emptied.
            self.last_loaded.forget(instance_id)
        return operation

    def restore_last(self) -> Operation | None:
        """Put back whatever was on the card before the manager stopped.

        Does nothing while something is already running. On Linux systemd owns
        the engines and they survive a manager restart — that is the reason for
        using it — so a manager coming back finds its model still answering.
        Only a machine that rebooted has anything to put back.

        Returns the operation, or None when there was nothing to do. Never
        raises: this runs while the manager is starting, and a model that
        cannot be restored must not stop the manager from serving.
        """
        if not self.last_loaded:
            return None
        remembered = self.last_loaded.all()
        if not remembered:
            return None
        config = self.store.load()
        if any(self.host.status(item.id).running for item in config.instances):
            return None
        # In the order they were loaded, stopping at the first that will not
        # go on. A machine given less memory than it had, or a reserve raised
        # since, must not be filled past what it can hold just because it once
        # held it — and the oldest was there first, so it is the one to keep.
        last = None
        for item in remembered:
            try:
                last = self.load(item["instance_id"], item["settings"] or None)
            except Exception as error:                  # reported, not raised
                self._log(f"Could not restore {item['instance_id']}: {error}")
                break
        return last

    def _log(self, text: str) -> None:
        self.bus.publish(LogEvent(source="restore", stream="err", text=text))

    def logs(self, instance_id: str, lines: int = 200) -> dict:
        """What the engine has printed about itself.

        Read only while the model is running. A stopped instance has a log on
        Linux, where systemd keeps the journal after the unit exits, and none
        on macOS, where the file belongs to a process that is gone — so a page
        offering it for a stopped model would work on one machine and not the
        other. Whether it *would not start* is a different question, answered
        by the sentence a failed load already carries.
        """
        instance = self.store.load().instance(instance_id)   # raises if unknown
        if not self.host.status(instance.id).running:
            return {"id": instance_id, "running": False, "lines": []}
        return {"id": instance_id, "running": True,
                "lines": self.host.logs(instance_id, lines=lines)}

    # -- configuring instances ---------------------------------------------

    def suggest_port(self) -> int:
        """The first free port at or above 8080.

        Offered when adding a model so there is one less thing to think about,
        and still editable, because a port sometimes has to match what a client
        already expects.
        """
        config = self.store.load()
        # The manager's own port counts as taken. An engine started on it would
        # find the port already held and refuse, which is a confusing way to
        # learn that the number was never free — and it is the number this
        # method hands out as soon as the instances reach it.
        taken = {item.port for item in config.instances} | {config.port}
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
        """Add an entry. The id is given rather than worked out.

        It used to be made from a label by lowercasing it and turning
        everything else into hyphens, which meant the name a request had to
        carry was decided by a sentence somebody wrote for reading. Now it is
        typed, checked, and is the only name the entry has.
        """
        engine = self.engines.get(payload["engine"])
        params = validate(engine.params(), payload.get("params", {}))
        identifier = str(payload.get("id", "")).strip()
        if not INSTANCE_ID.match(identifier):
            raise ValueError(
                "A name may hold lower-case letters, digits and hyphens, and "
                "must start with a letter or a digit. It is what a request "
                f"carries, so it has no spaces in it. {identifier!r} does not "
                "fit.")
        instance = Instance(
            id=identifier, engine=payload["engine"],
            model_id=payload["model_id"],
            port=int(payload["port"]), params=params,
        )
        with self.store.mutate() as config:
            if any(item.id == instance.id for item in config.instances):
                raise ValueError(
                    f"There is already a model called {instance.id}. The name "
                    f"is what a request asks for, so two cannot share one.")
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
        unknown = set(changes) - CHANGEABLE
        if unknown:
            # Silence here is worse than a refusal. Ignoring a field and still
            # answering "applied" tells the caller the change was made when
            # nothing happened.
            raise ValueError(
                f"Cannot change {', '.join(sorted(unknown))}. "
                f"Changeable: {', '.join(sorted(CHANGEABLE))}")
        with self.store.mutate() as config:
            instance = config.instance(instance_id)
            engine = self.engines.get(instance.engine)
            if "params" in changes:
                instance.params = validate(engine.params(), changes["params"])
            if "model_id" in changes:
                instance.model_id = str(changes["model_id"])
            if "port" in changes:
                port = int(changes["port"])
                if any(item.id != instance_id and item.port == port
                       for item in config.instances):
                    raise ValueError(f"Port {port} is already taken by another model")
                if port == config.port:
                    raise ValueError(f"Port {port} is the manager's own port")
                instance.port = port
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

    def build_status(self) -> list[dict]:
        return self.builds.all() if self.builds else []

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
        self._changed("models")
        return {"models_root": self.store.load().models_root}

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
        users = [item.id for item in config.instances if item.model_id == model_id]
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

    def search(self, query: str) -> dict:
        """Repositories holding something this machine can run.

        Only those. There was a switch to see the rest, and no answer to what
        it was for: a machine with no engine that reads safetensors cannot be
        helped by a list of them.

        `hidden` is what the filter took away, and it is the one thing the
        switch was good for. Nothing found and nothing *usable* found are
        different answers, and a list of length zero cannot tell them apart.
        """
        results = self.huggingface.search(query)
        supported = set(self.supported_formats())
        usable = [item for item in results
                  if supported.intersection(item["formats"])]
        return {"results": usable, "hidden": len(results) - len(usable)}

    def remote_sets(self, repo: str) -> list[dict]:
        """What a repository holds that this machine can run."""
        supported = set(self.supported_formats())
        return [item.json() for item in self.huggingface.sets(repo)
                if item.format in supported]

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
                "missing": list(model.missing),
                "capabilities": sorted(model.capabilities)}



# A shared library carries the execute bit and cannot be launched. Measured in
# llama.cpp's build directory on the container: 125 executable files, 33 of
# them `.so` companions to the launchers beside them.
LIBRARIES = (".so", ".dylib", ".dll")


def _is_library(name: str) -> bool:
    lowered = name.lower()
    return any(lowered.endswith(end) or f"{end}." in lowered for end in LIBRARIES)
