"""Engines that are one of AI-Lab's own server scripts, run in an isolated Python.

Most engines here are the same thing seen from the manager: a Python
environment of the engine's own, one of the small HTTP servers under
`ai_lab/` (music, speech, audio, images), one kind of work, one request shape,
one request at a time. `IsolatedEngine` is that shared part, written once (the
Template Method pattern: a subclass says which script, which task and which
paths, and writes `plan`, the command line). `ConfiguredEngine` adds the
per-model options most of them read from the configuration
(`engine_settings.<id>.model_options`), and serves only the models named there.

The engine protocol itself is `base.Engine`; nothing outside this package
knows these classes exist.
"""

from __future__ import annotations

import inspect
from pathlib import Path

from ..hosts.command import which
from ..network import LOOPBACK
from ..types import Format, ModelSet, Task
from .base import LaunchPlan, ParamSpec, validate
from .probe import http_ok

# The `ai_lab` package, where the server scripts live, and the repository
# above it, which a script needs on its path to import `ai_lab`.
AI_LAB_PACKAGE = Path(__file__).resolve().parents[1]
REPOSITORY = AI_LAB_PACKAGE.parent
# An engine's own editor page listens this far above its API port.
UI_PORT_OFFSET = 10000
MIB = 1024 * 1024


def ui_port(port: int) -> str:
    """The port of the engine's own page, beside its API on `port`."""
    return str(port + UI_PORT_OFFSET)


def model_directory(model: ModelSet) -> str:
    """The checkpoint's folder: the entry point itself, or the folder holding that file."""
    entry = Path(model.entrypoint)
    return str(entry.parent if entry.is_file() else entry)


def script_env(offline: bool = True, **extra: str) -> dict[str, str]:
    """The environment a server script runs with.

    Unbuffered output (the manager reads the log as it is written), the
    repository on the path (the script imports `ai_lab`), and, unless
    `offline` is false, Hugging Face kept off the network.
    """
    env = {"PYTHONUNBUFFERED": "1", "PYTHONPATH": str(REPOSITORY)}
    if offline:
        env["HF_HUB_OFFLINE"] = "1"
    return {**env, **extra}


def weights_mb(model: ModelSet) -> float:
    """The size of the model's files, in MiB."""
    return model.size_bytes / MIB


def with_headroom_mb(model: ModelSet, extra_mb: float) -> float:
    """The weights plus what a runtime holds beside them, whichever is larger:
    half as much again, or `extra_mb`. Weight bytes alone let the scheduler
    admit a model into less free memory than loading actually needs."""
    weights = weights_mb(model)
    return max(weights * 1.5, weights + extra_mb)


class IsolatedEngine:
    """One AI-Lab server script, one task, one request at a time.

    A subclass sets `id`, `display_name`, `task`, `paths` (the request shapes
    it answers) and `script` (the server, as parts of a path under `ai_lab/`),
    and writes `plan` — unless it is one of the shared audio or image servers'
    backends, in which case naming the `backend` is enough. `served_formats`,
    `settings` (the instance settings it accepts) and `finds_python` have
    defaults.
    """

    task: Task
    paths: tuple[str, ...] = ()
    script: tuple[str, ...] = ()
    served_formats: frozenset[Format] = frozenset({Format.SAFETENSORS})
    settings: tuple[ParamSpec, ...] = ()
    # Whether, with no binary configured, the Python on PATH is looked up
    # (shown on the Engines page) rather than the bare name `python`.
    finds_python: bool = False
    # The `--backend` of the shared server script, for engines that are one.
    backend: str = ""
    display_name: str

    def __init__(self, binary: str | None = None, server: str | None = None) -> None:
        self.binary = binary or (which("python") if self.finds_python else None) or "python"
        self.server = server or str(AI_LAB_PACKAGE.joinpath(*self.script))

    def formats(self) -> frozenset[Format]:
        """Weight formats this engine can load."""
        return self.served_formats

    def tasks(self) -> frozenset[Task]:
        """The one kind of work this engine does."""
        return frozenset({self.task})

    def params(self, task: Task | None = None) -> tuple[ParamSpec, ...]:
        """The instance settings this engine accepts for `task`."""
        return self.settings if (task or self.task) in self.tasks() else ()

    def plan(self, model: ModelSet, port: int, params: dict) -> LaunchPlan:
        """The shared server with this engine's backend, for this model."""
        if not self.backend:
            raise NotImplementedError(f"{type(self).__name__} must say how it starts")
        self._checked(model, params)
        return LaunchPlan(
            argv=[self.binary, self.server, "--backend", self.backend,
                  "--model", model.entrypoint, "--name", model.name,
                  "--host", LOOPBACK, "--port", str(port)],
            env={"PYTHONUNBUFFERED": "1"})

    def ready(self, port: int) -> bool:
        """True once the server answers its health check."""
        return http_ok(port)

    def concurrency(self, params: dict) -> int:
        """One request at a time: these servers hold one model and run it serially."""
        return 1

    def needs_mb(self, model: ModelSet, params: dict, card_total_mb: float) -> float:
        """The model's files, as an estimate of what it takes on the accelerator."""
        return weights_mb(model)

    def api_paths(self, task: Task | None = None) -> tuple[str, ...]:
        """The request shapes this engine answers for `task`."""
        return self.paths if (task or self.task) in self.tasks() else ()

    def _checked(self, model: ModelSet, params: dict) -> dict:
        """The instance settings, validated, after refusing a model of the wrong kind."""
        if model.format not in self.formats():
            raise ValueError(f"{self.display_name} cannot load {model.format.value} models")
        if model.task not in self.tasks():
            raise ValueError(f"{self.display_name} cannot perform {model.task.value}")
        return validate(self.settings, params)


class ConfiguredEngine(IsolatedEngine):
    """An isolated engine that serves only the models named in its configuration.

    `model_options` maps a model's name to what that model needs (a workflow,
    a voice, a memory reservation). When `reserves_memory` is true the
    scheduler is told each model's `memory_reservation_mb` instead of its
    file size, because the runtime holds far more than the weights.

    `location_settings` names the paths (and programs) the engine reads from
    its configuration section as flat keys — `output_root`, `cantor` — kept
    together in `locations` rather than one constructor argument each;
    `from_settings` reads them.
    """

    reserves_memory: bool = False
    location_settings: tuple[str, ...] = ()

    def __init__(self, binary: str | None = None, server: str | None = None,
                 model_options: dict[str, dict] | None = None,
                 locations: dict[str, str] | None = None) -> None:
        super().__init__(binary, server)
        self.model_options = dict(model_options or {})
        given = locations or {}
        self.locations = {name: given.get(name) or "" for name in self.location_settings}

    @classmethod
    def from_settings(cls, settings: dict) -> ConfiguredEngine:
        """Built from the engine's configuration section.

        The constructor's own arguments by name, and the `location_settings`
        gathered into `locations`. Keys the engine does not read (`source`,
        `update`, which belong to its installer) are left out.
        """
        accepted = inspect.signature(cls).parameters
        values = {key: value for key, value in settings.items()
                  if key in accepted and key != "locations"}
        if "locations" in accepted:
            values["locations"] = {name: settings.get(name) or ""
                                   for name in cls.location_settings}
        return cls(**values)

    def supports(self, model: ModelSet) -> bool:
        """Whether this is a model of the right kind that the configuration names."""
        return (model.format in self.formats() and model.task in self.tasks()
                and self.configured(model.name))

    def configured(self, model_name: str) -> bool:
        """Whether the configuration has options for this model."""
        return model_name in self.model_options

    def needs_mb(self, model: ModelSet, params: dict, card_total_mb: float) -> float:
        """The configured reservation, or the files' size when there is none to read."""
        if self.reserves_memory:
            return float(self.model_options[model.name]["memory_reservation_mb"])
        return weights_mb(model)

    def _options(self, model: ModelSet, params: dict, missing: str = "") -> dict:
        """This model's options, after refusing one the configuration does not name.

        Settings per instance are refused too: everything such an engine needs
        comes from its configuration. `missing` is the message when one of the
        `location_settings` is not configured.
        """
        if not self.supports(model):
            raise ValueError(f"{self.display_name} is not configured for {model.name}")
        if params:
            raise ValueError(f"{self.display_name} has no instance settings")
        if missing and not all(self.locations.values()):
            raise ValueError(missing)
        return self.model_options.get(model.name, {})
