"""Configured instance lifecycle and runtime transitions."""
from __future__ import annotations

from dataclasses import asdict, replace

from ..catalog import Catalog
from ..config import ConfigStore, INSTANCE_ID, Instance
from ..engines.base import validate
from ..hosts.base import Host
from ..runtime import Operation, Runtime
from ..types import LogEvent, Task

FIRST_PORT = 8080
CHANGEABLE = frozenset({"params", "model_id", "port"})


class InstanceService:
    def __init__(self, store: ConfigStore, catalog: Catalog,
                 runtime: Runtime, host: Host, engines, changed,
                 models, bus=None, last_loaded=None) -> None:
        self.store = store
        self.catalog = catalog
        self.runtime = runtime
        self.host = host
        self.engines = engines
        self._changed = changed
        self.models = models
        self.bus = bus
        self.last_loaded = last_loaded

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
                         "task": self._task(config, item.model_id),
                         "params": self._effective(
                             engine, item.params,
                             Task(self._task(config, item.model_id)))})
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
            task = Task(self._task(config, item.model_id))
            row["params"] = self._effective(engine, item.params, task)
            row["task"] = task.value
            if task is Task.SPEECH_SYNTHESIS and hasattr(engine, "speech_form"):
                row["speech_form"] = engine.speech_form(
                    item.model_id.rsplit("/", 1)[-1])
            rows.append(row)
        return rows

    def instance(self, instance_id: str) -> dict:
        """One configured entry, without asking what it is doing. See `configured`."""
        config = self.store.load()
        item = config.instance(instance_id)     # raises if unknown
        engine = self.engines.get(item.engine)
        return {"id": item.id, "engine": item.engine,
                "model_id": item.model_id, "port": item.port,
                "task": self._task(config, item.model_id),
                "params": self._effective(
                    engine, item.params,
                    Task(self._task(config, item.model_id)))}

    @staticmethod
    def _task(config, model_id: str) -> str:
        """The job of a model without walking its files."""
        repository_id = model_id.split("/", 1)[0]
        return config.repository(repository_id).task

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
    def _effective(engine, stored: dict,
                   task: Task = Task.TEXT_GENERATION) -> dict:
        try:
            specs = engine.params(task)
            return validate(specs, {key: value for key, value in stored.items()
                                    if key in {spec.key for spec in specs}})
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
        config = self.store.load()
        instance = config.instance(instance_id)
        engine = self.engines.get(instance.engine)
        task = Task(self._task(config, instance.model_id))
        return validate(engine.params(task), {**instance.params, **settings})

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
        models = self.models()
        engines = self.engines.describe(capabilities)
        for description in engines:
            engine = self.engines.get(description["id"])
            configured_names = getattr(engine, "model_configs", None)
            if configured_names is None:
                configured_names = getattr(engine, "model_modes", None)
            if configured_names is None:
                configured_names = getattr(engine, "model_options", None)
            if configured_names is not None:
                description["supported_model_ids"] = [
                    model["id"] for model in models
                    if model["name"] in configured_names
                    and model["format"] in description["formats"]
                    and model["task"] in description["tasks"]]
        return {"port": self.suggest_port(), "engines": engines,
                "models": models}

    def create_instance(self, payload: dict) -> dict:
        """Add an entry. The id is given rather than worked out.

        It used to be made from a label by lowercasing it and turning
        everything else into hyphens, which meant the name a request had to
        carry was decided by a sentence somebody wrote for reading. Now it is
        typed, checked, and is the only name the entry has.
        """
        config = self.store.load()
        engine = self.engines.get(payload["engine"])
        model = self.catalog.find(config.repositories, payload["model_id"])
        if hasattr(engine, "supports") and not engine.supports(model):
            raise ValueError(f"{engine.display_name} is not configured for {model.name}")
        params = validate(engine.params(model.task), payload.get("params", {}))
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
            target_model_id = str(changes.get("model_id", instance.model_id))
            model = self.catalog.find(config.repositories, target_model_id)
            if hasattr(engine, "supports") and not engine.supports(model):
                raise ValueError(f"{engine.display_name} is not configured for {model.name}")
            if "params" in changes:
                instance.params = validate(engine.params(model.task), changes["params"])
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

    def _resolve(self, instance_id: str):
        config = self.store.load()
        instance = config.instance(instance_id)
        model = self.catalog.find(config.repositories, instance.model_id)
        return instance, model
