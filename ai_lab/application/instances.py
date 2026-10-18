"""Configured instance lifecycle and runtime transitions."""
from __future__ import annotations

from dataclasses import replace

from ..engines.base import validate
from ..runtime.core import Operation
from ..types import LogEvent, Task
from .instance_config import InstanceConfiguration, InstanceParts

# `InstanceParts` is part of this service's interface: callers import it from here.
__all__ = ["InstanceParts", "InstanceService"]


class InstanceService:
    """The configured model entries: listing them, loading and unloading, editing.

    Editing is its `InstanceConfiguration`; this class lists, loads, unloads
    and restores. It keeps no state that needs a lock: the configuration file
    is guarded by `ConfigStore`, the processes by the host.
    """

    def __init__(self, parts: InstanceParts) -> None:
        self.store = parts.store
        self.catalog = parts.catalog
        self.runtime = parts.runtime
        self.host = parts.host
        self.engines = parts.engines
        self._changed = parts.changed
        self.models = parts.models
        self.bus = parts.bus
        self.last_loaded = parts.last_loaded
        # Editing the entries: suggesting a port, the Add form, create,
        # change, delete. Callers reach it as `instance_service.configuration`.
        self.configuration = InstanceConfiguration(parts)

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
            if task is Task.MUSIC_GENERATION and hasattr(engine, "music_form"):
                row["music_form"] = engine.music_form(
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
        """Stop this model; the card is no longer remembered as holding it."""
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

    def apply_and_reload(self, instance_id: str, changes: dict) -> dict:
        """Save the settings and restart the model with them.

        The two halves are one action here because that is what the user
        means: the settings decide how much is reserved on the accelerator, so
        they only take effect when the model starts again.
        """
        self.configuration.update_instance(instance_id, changes)
        operation = self.load(instance_id)
        return {"id": instance_id, "applied": operation.ok,
                "operation": operation.json()}

    def _resolve(self, instance_id: str):
        config = self.store.load()
        instance = config.instance(instance_id)
        model = self.catalog.find(config.repositories, instance.model_id)
        return instance, model
