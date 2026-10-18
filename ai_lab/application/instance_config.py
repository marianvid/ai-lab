"""Adding, changing and deleting configured model entries.

A base class of `InstanceService` (a mixin): everything that edits the
`instances` section of the configuration. Saving never restarts a model —
applying means restarting, and restarting unloads a model somebody may be
using, so that is a separate act (`InstanceService.apply_and_reload`).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
from typing import Any

from ..catalog.core import Catalog
from ..config.store import INSTANCE_ID, ConfigStore, Instance
from ..engines.base import validate
from ..hosts.base import Host

FIRST_PORT = 8080
CHANGEABLE = frozenset({"params", "model_id", "port"})
# Where an engine lists which model names it is configured for, in order of preference.
CONFIGURED_NAMES = ("model_configs", "model_modes", "model_options")


class InstanceConfiguration:
    """The entries' configuration (see `InstanceService`, which owns the fields)."""

    store: ConfigStore
    catalog: Catalog
    host: Host
    engines: Any
    models: Callable[[], list[dict]]
    _changed: Callable[[str], None]

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
            supported = _supported_model_ids(self.engines.get(description["id"]),
                                             description, models)
            if supported is not None:
                description["supported_model_ids"] = supported
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
                instance.port = _free_port(config, instance_id, int(changes["port"]))
        self._changed("instances")
        running = self.host.status(instance_id).running
        return {"id": instance_id, "applied": not running,
                "note": "" if not running
                        else "Reload to apply the new settings"}

    def delete_instance(self, instance_id: str) -> None:
        """Remove a stopped entry from the configuration."""
        if self.host.status(instance_id).running:
            raise ValueError("Stop the instance before deleting it")
        with self.store.mutate() as config:
            config.instance(instance_id)                # raises if unknown
            config.instances = [item for item in config.instances
                                if item.id != instance_id]
        self._changed("instances")


def _supported_model_ids(engine, description: dict, models: list[dict]) -> list[str] | None:
    """The models an engine that names its models is configured for; None when it names none."""
    configured_names = next((names for names in
                             (getattr(engine, attribute, None) for attribute in CONFIGURED_NAMES)
                             if names is not None), None)
    if configured_names is None:
        return None
    return [model["id"] for model in models
            if model["name"] in configured_names
            and model["format"] in description["formats"]
            and model["task"] in description["tasks"]]


def _free_port(config, instance_id: str, port: int) -> int:
    """`port`, after checking no other entry and not the manager itself uses it."""
    if any(item.id != instance_id and item.port == port for item in config.instances):
        raise ValueError(f"Port {port} is already taken by another model")
    if port == config.port:
        raise ValueError(f"Port {port} is the manager's own port")
    return port
