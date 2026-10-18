"""Changes to this machine's settings from the Settings page.

Each change is checked before it is saved: a number that cannot work, or a
path that is not there, is refused while the person is still looking at the
field they typed into, not discovered later when a request hangs or every
screen shows an engine as not installed.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

from .. import budget
from ..config import ConfigStore, ModelRoot, validate_distinct_roots
from ..hosts.base import Host
from ..settings import Settings

# The front door's own settings, with the range each may take.
GATEWAY_SETTINGS: dict[str, tuple[float, float]] = {
    "first_byte_s": (1.0, 3600.0),
    "between_bytes_s": (1.0, 3600.0),
    "max_waiting": (1, 10000),
}
# The most that may be held back for the machine itself: 1 TB, in MB.
MAX_RESERVE_MB = 1024 * 1024
MODEL_ROOT_FIELDS = frozenset({"path", "enabled", "writable", "download_default"})


class SettingsChanges:
    """The gateway's limits, the memory held back, engine programs and model roots."""

    def __init__(self, store: ConfigStore, host: Host, settings: Settings,
                 changed: Callable[[str], None]) -> None:
        self.store = store
        self.host = host
        self.settings = settings
        self._changed = changed

    # -- the front door ----------------------------------------------------

    def gateway_settings(self) -> dict:
        """How long the front door waits and how many it holds, as saved."""
        return dict(self.store.load().gateway)

    def update_gateway(self, changes: dict) -> dict:
        """Change how long the front door waits and how many it holds.

        Checked here rather than in the gateway, for the same reason every
        other setting is checked outside the thing it configures.
        """
        _refuse_unknown(changes, set(GATEWAY_SETTINGS))
        cleaned = {key: _in_range(key, value, *GATEWAY_SETTINGS[key])
                   for key, value in changes.items()}
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
        _refuse_unknown(changes, {"reserve_mb"})
        cleaned = {}
        if "reserve_mb" in changes:
            cleaned["reserve_mb"] = _reserve(changes["reserve_mb"])
        with self.store.mutate() as config:
            config.memory = {**config.memory, **cleaned}
        self._changed("settings")
        return self.memory_budget()

    # -- engines and where models live --------------------------------------

    def update_engine_binary(self, engine_id: str, path: str) -> dict:
        """Point an engine at a different program.

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

    def update_model_root(self, root_id: str, changes: dict) -> None:
        """Change one model root: its path, whether it is used, written to, downloaded to."""
        unknown = set(changes) - MODEL_ROOT_FIELDS
        if unknown:
            raise ValueError(
                f"Unknown model-root settings: {', '.join(sorted(unknown))}")
        with self.store.mutate() as config:
            model_root = config.model_root(root_id)
            if "path" in changes:
                model_root.path = _directory(changes["path"])
                if root_id == "core":
                    config.models_root = model_root.path
            for flag in ("enabled", "writable"):
                if flag in changes:
                    setattr(model_root, flag, bool(changes[flag]))
            config.download_root = _download_root(
                config.download_root, root_id, model_root, changes.get("download_default"))
            validate_distinct_roots(config.model_roots)
        self._changed("settings")
        self._changed("models")

    def update_repository(self, repository_id: str, changes: dict) -> dict:
        """Rename a repository, or say whether it may be written to.

        Its path is not among these: it comes from the models root and this
        repository's id. See `update_models_root`.
        """
        unknown = set(changes) - {"name", "writable"}
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


def _refuse_unknown(changes: dict, known: set[str]) -> None:
    unknown = set(changes) - known
    if unknown:
        raise ValueError(f"Unknown settings: {', '.join(sorted(unknown))}")


def _in_range(key: str, value, low: float, high: float) -> float:
    """`value` as a number of `low`'s kind, between `low` and `high`."""
    try:
        number = int(value) if isinstance(low, int) else float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{key} must be a number") from None
    if not low <= number <= high:
        raise ValueError(f"{key} must be between {low} and {high}")
    return number


def _reserve(value) -> float:
    """The memory to hold back, in MB: a number from 0 to `MAX_RESERVE_MB`."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError("reserve_mb must be a number") from None
    if not 0 <= number <= MAX_RESERVE_MB:
        raise ValueError(f"reserve_mb must be between 0 and {MAX_RESERVE_MB}")
    return number


def _directory(text) -> str:
    """An existing directory, resolved; ValueError when it is not one."""
    path = Path(str(text)).expanduser()
    if not path.is_dir():
        raise ValueError(f"{path} is not a directory")
    return str(path.resolve())


def _download_root(current: str, root_id: str, model_root: ModelRoot,
                   download_default) -> str:
    """Where downloads go after this change to one root's `download_default`."""
    if download_default:
        if not model_root.enabled:
            raise ValueError("A disabled model root cannot receive downloads")
        return root_id
    if download_default is False and current == root_id:
        return "core"
    return current
