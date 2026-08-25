"""Shared data structures.

These describe *what things are*, with no knowledge of disk, network or GPU.
They exist here rather than inside a service so that catalog, runtime and
downloads can all speak about the same model without importing one another.

Nothing in this file performs I/O. If a type here needs to read a file, it
belongs in a service instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Format(str, Enum):
    """Weight format on disk. Determines which engines can serve a model."""

    GGUF = "gguf"
    SAFETENSORS = "safetensors"
    FP8 = "fp8"
    NVFP4 = "nvfp4"
    AWQ = "awq"
    GPTQ = "gptq"
    NEMO = "nemo"
    ONNX = "onnx"
    PYANNOTE = "pyannote"


class Task(str, Enum):
    """What a model does, independently of how its weights are stored."""

    TEXT_GENERATION = "text-generation"
    TRANSCRIPTION = "transcription"
    ALIGNMENT = "alignment"
    VAD = "vad"
    DIARIZATION = "diarization"


@dataclass(frozen=True, slots=True)
class ModelFile:
    """One file on disk belonging to a model."""

    path: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class ModelSet:
    """A complete model: every file needed to load it, as one unit.

    A model is frequently split into numbered shards, for example
    `model-00001-of-00005.safetensors`, alongside a tokenizer and a config
    file. All of them are required — a missing shard makes the model
    unloadable — so the whole group is treated as a single object everywhere
    in the application.

    `entrypoint` is the path handed to the engine. For GGUF that is the first
    shard (llama.cpp finds the rest itself); for safetensors it is the
    directory.
    """

    id: str
    name: str
    format: Format
    entrypoint: str
    files: tuple[ModelFile, ...]
    task: Task = Task.TEXT_GENERATION
    complete: bool = True
    missing: tuple[str, ...] = ()
    # What the weights can do, read from the model's own files rather than
    # configured: whether it can read pictures, and whether it has been taught
    # to ask for tools. An entry can take these away; it cannot add them.
    capabilities: frozenset[str] = frozenset()

    @property
    def size_bytes(self) -> int:
        return sum(item.size_bytes for item in self.files)


@dataclass(frozen=True, slots=True)
class Capabilities:
    """What the machine we are running on can actually do.

    The UI reads this to decide what to disable rather than guessing from the
    operating system name.
    """

    supervisor: str               # "systemd" or "subprocess"
    engines: frozenset[str]       # engine ids usable here
    accelerator_kind: str         # "cuda", "metal" or "none"
    can_configure_accelerator: bool = False
    operating_system: str = "unknown"


@dataclass(frozen=True, slots=True)
class AcceleratorSnapshot:
    """A reading of the GPU at one moment.

    `memory_kind` distinguishes a discrete card from Apple's unified memory.
    On unified memory there is no separate VRAM pool, so a progress bar drawn
    from these numbers means something different — the UI needs to know which.
    """

    available: bool
    name: str
    kind: str                     # "cuda", "metal", "none"
    memory_kind: str              # "dedicated" or "unified"
    memory_used_mb: float = 0.0   # everything on the accelerator
    memory_total_mb: float = 0.0
    process_memory_mb: float = 0.0  # just the process that was asked about
    temperature_c: float | None = None
    utilization_percent: float | None = None


@dataclass(frozen=True, slots=True)
class ProcessSpec:
    """Everything needed to launch one engine process."""

    instance_id: str
    argv: list[str]
    env: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ProcessStatus:
    running: bool
    pid: int | None = None


class Phase(str, Enum):
    """Steps of a load or unload, in order.

    Used both to drive the progress display and to record how long each step
    took.
    """

    STOPPING = "stopping"
    PROCESS_GONE = "process_gone"
    MEMORY_RELEASED = "memory_released"
    STARTING = "starting"
    PROCESS_UP = "process_up"
    WEIGHTS_LOADING = "weights_loading"
    READY = "ready"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class LogEvent:
    """One line of output from a long-running job, streamed to the browser.

    Separate from RuntimeEvent because it answers a different question: that
    one reports where a model load has got to, this one carries the text a
    build is printing. Both travel on the same bus so the browser needs one
    connection.
    """

    source: str                   # what produced it, e.g. "llamacpp"
    stream: str                   # "out", "err" or "status"
    text: str
    elapsed_ms: int = 0


@dataclass(frozen=True, slots=True)
class ChangeEvent:
    """Something changed; whoever is watching may want to look again.

    Sent instead of having the browser ask every few seconds. Polling redrew
    the page whether or not anything had happened, which threw away whatever
    was half-typed or selected at the time — and the quieter the machine, the
    more pointless the redraw.

    It carries only what changed, not the new value: the page asks for that
    itself, which keeps this from becoming a second, competing description of
    the state.
    """

    topic: str                    # instances, downloads, models, engines, storage


@dataclass(frozen=True, slots=True)
class RuntimeEvent:
    """One progress update, streamed to the browser.

    `progress` is what the bar draws: how far along this operation is, from
    nothing to finished. It always runs 0 to 1, whichever way the memory is
    moving and however large the model is.

    The memory figures are reported separately, as numbers rather than as the
    bar. They answer a different question — how much of the card this model
    occupies — and using them as the bar would mean a 4 GB model on a 32 GB
    card looked 13% loaded when it was in fact ready.

    `memory_used_mb` is this instance alone; `accelerator_used_mb` is the whole
    card.
    """

    instance_id: str
    phase: Phase
    elapsed_ms: int
    progress: float               # 0.0 to 1.0 — how far this operation has got
    memory_used_mb: float         # this instance alone
    memory_total_mb: float
    accelerator_used_mb: float = 0.0   # every model on the card, for context
    message: str = ""


@dataclass(frozen=True, slots=True)
class Change:
    """One thing that would change if an engine were updated.

    Deliberately not tied to where it came from. llama.cpp is a git checkout,
    so a change there is a commit; vLLM is installed as a package, so a change
    there is a paragraph from a release note. Both arrive here as the same
    three fields, and the interface draws them the same way.

    `area` is the part of the engine it touches — "CUDA", "server", "new
    models" — used to sort what this machine cares about from what belongs to
    somebody else's platform. `reference` is whatever identifies it upstream:
    a commit hash, a pull request number, a version.
    """

    area: str
    title: str
    reference: str = ""


@dataclass(frozen=True, slots=True)
class Interests:
    """What this machine actually uses, so an update can be read against it.

    Every field is worked out from the machine and from what is configured on
    it, never written down by hand. The Mac cares about Metal and the container
    cares about CUDA, and neither should be told which it is.

    `pictures` is true when some configured entry points at a model whose
    weights can read them. Somebody running only text models does not need to
    read a hundred lines about the vision code.
    """

    accelerator_kind: str         # "cuda", "metal" or "none"
    formats: frozenset[str] = frozenset()      # "gguf", "nvfp4", …
    pictures: bool = False
    tools: bool = False


@dataclass(frozen=True, slots=True)
class Changes:
    """What an update would bring, sorted into what matters here and what does not.

    `yours` and `others` together are everything found: nothing is thrown away,
    because a summary that quietly drops things is one nobody can trust. The
    interface shows `yours` and keeps `others` behind a count.

    `notes` is prose written by the people who made the release, when there is
    any. It is better than any list of commits and is shown above them.

    `unreadable` says why this is incomplete rather than pretending it is not —
    no network, a checkout that git cannot read, a version with no release.
    """

    installed: str
    latest: str
    yours: tuple[Change, ...] = ()
    others: tuple[Change, ...] = ()
    notes: str = ""
    unreadable: str = ""
