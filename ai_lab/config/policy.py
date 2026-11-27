"""Typed runtime policies read from the versioned JSON configuration.

The JSON remains a portable, editable mapping. These immutable values are the
boundary used by application services so defaults and limits have one owner.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Mapping

from ..types import Task


# The gateway's defaults. This module owns them; `gateway/limits.py` and
# `api/passthrough.py` read them from here rather than writing them again.
FIRST_BYTE_S = 120.0
BETWEEN_BYTES_S = 30.0
MAX_WAITING = 150
STALL_S = 60.0
SAFETY_S = 1800.0
DEFAULT_MAX_TOKENS = 4096

# The defaults of the `limits` section (`MachineLimits`; docs/settings.md, "Limits kept
# in the configuration").
START_S = 60.0
LOAD_S = 900.0
UNLOAD_S = 120.0
INSTALL_S = 3600.0
BUILD_S = 3600.0
UPDATE_CHECK_EVERY_S = 3600.0
BUILDS_FIRST_CHECK_S = 20.0
INSTALLS_FIRST_CHECK_S = 25.0
SYSTEM_PATH = "/usr/local/bin:/usr/bin:/bin"
# The range each of the gateway's own settings may take: one owner, read by the
# start and by the Gateway page's save alike (audit m20).
GATEWAY_BOUNDS: dict[str, tuple[int, int]] = {
    "first_byte_s": (1, 86400), "between_bytes_s": (1, 86400), "max_waiting": (1, 10000),
    "stall_s": (5, 3600), "safety_s": (60, 86400), "default_max_tokens": (0, 1_000_000),
}
EMBEDDINGS_START_S = 120.0
EMBEDDINGS_STOP_S = 30.0
EMBEDDINGS_DRAIN_S = 60.0
EMBEDDINGS_HOLD_S = 900.0
EMBEDDINGS_CALL_S = 600.0
FIRST_PORT = 8080
# An engine's own editor page listens this far above its API port.
EDITOR_PORT_OFFSET = 10000
CHARS_PER_TOKEN = 3.0
RETRY_AFTER_S = 5
# The longest any of those waits may be set to: a week.
LONGEST_S = 7 * 86400.0


def _number(raw: Mapping, name: str, default: float, *, bounds: tuple[float, float],
            section: str = "Gateway") -> float:
    minimum, maximum = bounds
    value = raw.get(name, default)
    if type(value) not in (int, float) or not minimum <= value <= maximum:
        raise ValueError(f"{section} {name} must be between {minimum:g} and {maximum:g}")
    return float(value)


def _integer(raw: Mapping, name: str, default: int, *, bounds: tuple[int, int],
             section: str) -> int:
    minimum, maximum = bounds
    value = raw.get(name, default)
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{section} {name} must be between {minimum} and {maximum}")
    return value


@dataclass(frozen=True, slots=True)
class TaskTimeout:
    """The two waits for one kind of request: to the first byte, and between bytes."""

    first_byte_s: float
    between_bytes_s: float


@dataclass(frozen=True, slots=True)
class GatewayPolicy:
    """The `gateway` section: how long to wait for an engine, how many to hold, upload limits."""

    first_byte_s: float = FIRST_BYTE_S
    between_bytes_s: float = BETWEEN_BYTES_S
    max_waiting: int = MAX_WAITING
    task_timeouts: dict[str, TaskTimeout] = field(default_factory=dict)
    max_upload_bytes: int = 0
    max_upload_pixels: int = 0
    max_upload_dimension: int = 0
    # For engines that report their progress (vLLM): see `gateway.limits.Patience`.
    stall_s: float = STALL_S
    safety_s: float = SAFETY_S
    default_max_tokens: int = DEFAULT_MAX_TOKENS

    @classmethod
    def from_mapping(cls, raw: Mapping | None) -> GatewayPolicy:
        """Read and check the section; ValueError names what is wrong."""
        raw = {} if raw is None else raw
        if not isinstance(raw, Mapping):
            raise ValueError("Gateway settings must be an object")
        first = _number(raw, "first_byte_s", FIRST_BYTE_S, bounds=GATEWAY_BOUNDS["first_byte_s"])
        between = _number(raw, "between_bytes_s", BETWEEN_BYTES_S,
                          bounds=GATEWAY_BOUNDS["between_bytes_s"])
        timeouts = raw.get("task_timeouts", {})
        if not isinstance(timeouts, dict):
            raise ValueError("Gateway task_timeouts must be an object")
        tasks = {task.value for task in Task}
        by_task = {}
        for task, values in timeouts.items():
            if task not in tasks or not isinstance(values, dict):
                raise ValueError(f"Gateway task_timeouts has invalid task {task!r}")
            by_task[task] = TaskTimeout(
                first_byte_s=_number(values, "first_byte_s", first,
                                     bounds=GATEWAY_BOUNDS["first_byte_s"]),
                between_bytes_s=_number(values, "between_bytes_s", between,
                                        bounds=GATEWAY_BOUNDS["between_bytes_s"]),
            )
        return cls(
            first_byte_s=first,
            between_bytes_s=between,
            max_waiting=_integer(raw, "max_waiting", MAX_WAITING,
                                 bounds=GATEWAY_BOUNDS["max_waiting"], section="Gateway"),
            task_timeouts=by_task,
            max_upload_bytes=_integer(raw, "max_upload_bytes", 0, bounds=(0, 1024 ** 3),
                                      section="Gateway"),
            max_upload_pixels=_integer(raw, "max_upload_pixels", 0, bounds=(0, 1024 ** 3),
                                       section="Gateway"),
            max_upload_dimension=_integer(raw, "max_upload_dimension", 0,
                                          bounds=(0, 100000), section="Gateway"),
            stall_s=_number(raw, "stall_s", STALL_S, bounds=GATEWAY_BOUNDS["stall_s"]),
            safety_s=_number(raw, "safety_s", SAFETY_S, bounds=GATEWAY_BOUNDS["safety_s"]),
            default_max_tokens=_integer(raw, "default_max_tokens", DEFAULT_MAX_TOKENS,
                                        bounds=GATEWAY_BOUNDS["default_max_tokens"],
                                        section="Gateway"),
        )

    def timeout_mapping(self) -> dict[str, dict[str, float]]:
        """The per-task waits as plain dictionaries, as the gateway takes them."""
        return {task: {"first_byte_s": value.first_byte_s,
                       "between_bytes_s": value.between_bytes_s}
                for task, value in self.task_timeouts.items()}


def _part(raw: Mapping, name: str) -> Mapping:
    """One sub-section of `limits`; an absent one is empty, anything but an object is refused."""
    part = raw.get(name, {})
    if not isinstance(part, Mapping):
        raise ValueError(f"Limits {name} must be an object")
    return part


def _seconds(raw: Mapping, name: str, default: float, section: str) -> float:
    return _number(raw, name, default, bounds=(1, LONGEST_S), section=section)


@dataclass(frozen=True, slots=True)
class RuntimeLimits:
    """How long each step of a load or an unload may take (`runtime/core.py`).

    `start_s`: for the engine's process to appear. `load_s`: for it to answer
    once started (weights read, a first compile). `unload_s`: for the process
    to exit, and again for its memory to come back.
    """

    start_s: float = START_S
    load_s: float = LOAD_S
    unload_s: float = UNLOAD_S

    @classmethod
    def from_mapping(cls, raw: Mapping) -> RuntimeLimits:
        """Read and check `limits.runtime`."""
        section = "Limits runtime"
        return cls(start_s=_seconds(raw, "start_s", START_S, section),
                   load_s=_seconds(raw, "load_s", LOAD_S, section),
                   unload_s=_seconds(raw, "unload_s", UNLOAD_S, section))


@dataclass(frozen=True, slots=True)
class BuildLimits:
    """Engine builds and installs (`builds/`): their time, their timers, their search path.

    `install_s`: one install command (a package install, a git application's
    step). `build_s`: one compile step of a source build. `check_every_s`:
    how often upstream is asked for new versions; the source builds ask first
    `builds_first_check_s` after start, the installs `installs_first_check_s`
    (apart, so the two do not ask at once). `system_path`: where the programs a
    new environment is built with are found.
    """

    install_s: float = INSTALL_S
    build_s: float = BUILD_S
    check_every_s: float = UPDATE_CHECK_EVERY_S
    builds_first_check_s: float = BUILDS_FIRST_CHECK_S
    installs_first_check_s: float = INSTALLS_FIRST_CHECK_S
    system_path: str = SYSTEM_PATH

    @classmethod
    def from_mapping(cls, raw: Mapping) -> BuildLimits:
        """Read and check `limits.builds`."""
        section = "Limits builds"
        path = raw.get("system_path", SYSTEM_PATH)
        if not isinstance(path, str) or not path or not all(
                part.startswith("/") for part in path.split(":")):
            raise ValueError(f"{section} system_path must be absolute folders joined by ':'")
        return cls(install_s=_seconds(raw, "install_s", INSTALL_S, section),
                   build_s=_seconds(raw, "build_s", BUILD_S, section),
                   check_every_s=_seconds(raw, "check_every_s", UPDATE_CHECK_EVERY_S, section),
                   builds_first_check_s=_seconds(raw, "builds_first_check_s",
                                                 BUILDS_FIRST_CHECK_S, section),
                   installs_first_check_s=_seconds(raw, "installs_first_check_s",
                                                   INSTALLS_FIRST_CHECK_S, section),
                   system_path=path)


@dataclass(frozen=True, slots=True)
class EmbeddingsLimits:
    """The embeddings lane's waits (`cpu_models/`).

    `start_s`: for a model's server to answer after it starts. `stop_s`: for
    it to exit when asked, before it is killed. `drain_s`: for the requests on
    the card to finish when the card is wanted back. `hold_s`: how long a model
    stays off the card after that. `call_s`: one embeddings call.
    """

    start_s: float = EMBEDDINGS_START_S
    stop_s: float = EMBEDDINGS_STOP_S
    drain_s: float = EMBEDDINGS_DRAIN_S
    hold_s: float = EMBEDDINGS_HOLD_S
    call_s: float = EMBEDDINGS_CALL_S

    @classmethod
    def from_mapping(cls, raw: Mapping) -> EmbeddingsLimits:
        """Read and check `limits.embeddings`."""
        section = "Limits embeddings"
        return cls(start_s=_seconds(raw, "start_s", EMBEDDINGS_START_S, section),
                   stop_s=_seconds(raw, "stop_s", EMBEDDINGS_STOP_S, section),
                   drain_s=_seconds(raw, "drain_s", EMBEDDINGS_DRAIN_S, section),
                   hold_s=_seconds(raw, "hold_s", EMBEDDINGS_HOLD_S, section),
                   call_s=_seconds(raw, "call_s", EMBEDDINGS_CALL_S, section))


@dataclass(frozen=True, slots=True)
class ClientCapLimits:
    """How a client's cap counts and refuses (`clients/caps.py`).

    `chars_per_token`: the prompt's tokens are estimated as its characters
    divided by this. `retry_after_s`: how long a client over its cap is told
    to wait.
    """

    chars_per_token: float = CHARS_PER_TOKEN
    retry_after_s: int = RETRY_AFTER_S

    @classmethod
    def from_mapping(cls, raw: Mapping) -> ClientCapLimits:
        """Read and check `limits.clients`."""
        section = "Limits clients"
        return cls(chars_per_token=_number(raw, "chars_per_token", CHARS_PER_TOKEN,
                                           bounds=(0.5, 20), section=section),
                   retry_after_s=_integer(raw, "retry_after_s", RETRY_AFTER_S,
                                          bounds=(1, 3600), section=section))


@dataclass(frozen=True, slots=True)
class MachineLimits:
    """The `limits` section: this machine's waits and ports, each with today's default.

    Every value may be left out; then the default holds. See
    docs/settings.md ("Limits kept in the configuration") for what each one is
    for.
    """

    runtime: RuntimeLimits = field(default_factory=RuntimeLimits)
    builds: BuildLimits = field(default_factory=BuildLimits)
    embeddings: EmbeddingsLimits = field(default_factory=EmbeddingsLimits)
    clients: ClientCapLimits = field(default_factory=ClientCapLimits)
    first_port: int = FIRST_PORT
    editor_port_offset: int = EDITOR_PORT_OFFSET

    @classmethod
    def from_mapping(cls, raw: Mapping | None) -> MachineLimits:
        """Read and check the section; ValueError names what is wrong."""
        raw = {} if raw is None else raw
        if not isinstance(raw, Mapping):
            raise ValueError("Limits must be an object")
        return cls(runtime=RuntimeLimits.from_mapping(_part(raw, "runtime")),
                   builds=BuildLimits.from_mapping(_part(raw, "builds")),
                   embeddings=EmbeddingsLimits.from_mapping(_part(raw, "embeddings")),
                   clients=ClientCapLimits.from_mapping(_part(raw, "clients")),
                   first_port=_integer(raw, "first_port", FIRST_PORT, bounds=(1, 65535),
                                       section="Limits"),
                   editor_port_offset=_integer(raw, "editor_port_offset", EDITOR_PORT_OFFSET,
                                               bounds=(1000, 50000), section="Limits"))
