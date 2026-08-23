"""Reading and writing config.json.

Only this file knows the on-disk shape of the configuration. Saving is atomic
— write a temporary file, then rename — so a crash mid-write cannot leave a
truncated configuration behind.

Engine settings are nested under `params` rather than spread across the
instance record, so adding an engine never changes this schema:

    {
      "repositories": [
        {"id": "gguf", "name": "GGUF", "path": "/models/gguf", "format": "gguf"}
      ],
      "instances": [
        {"id": "qwen", "name": "Coding", "engine": "llamacpp",
         "model_id": "gguf/qwen-coder/Qwen3.6-35B", "port": 8080,
         "params": {"context_size": 32768}}
      ]
    }

Changing configuration goes through `mutate()`, which reads, hands you the
object, then writes it back under a lock. That keeps read-modify-write in one
place instead of scattering it across services.
"""

from __future__ import annotations

import json
import re
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from threading import RLock
from typing import Iterator


@dataclass(slots=True)
class Repository:
    """A directory holding models of one weight format.

    The format is declared rather than guessed: FP8, NVFP4, AWQ and GPTQ are
    all `.safetensors` files, so the extension cannot tell them apart. The
    layout on disk keeps one format per directory precisely so this
    declaration is enough.
    """

    id: str
    name: str
    format: str
    # Where this format's models are, worked out rather than stored: the models
    # root plus the format. One root is set and the rest follows, so GGUF and
    # NVFP4 cannot end up on different disks by accident, and a machine set up
    # from this configuration has the same tree as every other.
    #
    # The *format* names the folder, not the id. `MODEL_STORAGE.md` has
    # described a format-first tree all along — `gguf/`, `safetensors/`,
    # `fp8/`, `nvfp4/` — and an id is a name somebody chose, which may be
    # shorter.
    path: str = ""
    writable: bool = True


# What an id may be made of. Letters, digits and hyphens, which is what fits in
# a URL, in a request body and in an environment variable without quoting — and
# what stays the same wherever it is written down.
INSTANCE_ID = re.compile(r"^[a-z0-9][a-z0-9-]*$")


@dataclass(slots=True)
class Instance:
    """One configured engine slot. Not necessarily running.

    The id is the only name it has. There used to be a label beside it — a
    descriptive sentence for reading — and it turned into a second way of
    naming the same thing, which had to be kept in step with the first and was
    the one people saw while the other was the one that worked.

    So the id carries both jobs: it is what a request names, and it is what a
    person reads. That is why it is typed rather than derived, and why it is
    the only thing here that has to be unique.
    """

    id: str
    engine: str
    model_id: str
    port: int
    params: dict = field(default_factory=dict)


@dataclass(slots=True)
class Config:
    title: str = "AI-Lab"
    host: str = "0.0.0.0"
    port: int = 8090
    repositories: list[Repository] = field(default_factory=list)
    instances: list[Instance] = field(default_factory=list)
    # Per-engine overrides, keyed by engine id. Today the only setting is
    # `binary`, the path to the executable. It is worth configuring rather
    # than resolving from PATH: a machine can easily hold two builds of
    # llama.cpp — a packaged one and your own — and PATH silently picks
    # whichever comes first, which may not be the one you tuned.
    engines: dict = field(default_factory=dict)
    # The front door's own settings: how long to wait for an engine, and how
    # many requests to hold. They belong to the machine rather than to a
    # request — the right numbers on a card that reads 8,400 tokens of prompt
    # in under a second are not the right numbers on a Mac running a 70 GB
    # model at 17 tokens a second — so they are configured, not sent.
    gateway: dict = field(default_factory=dict)
    # How much of this machine's memory to leave for the machine. See
    # `budget.py` for why it is a reserve rather than an allowance.
    memory: dict = field(default_factory=dict)
    # The one directory every model lives under. Each repository is a
    # folder in it, named after the format.
    models_root: str = ""

    def repository(self, repository_id: str) -> Repository:
        found = next((item for item in self.repositories if item.id == repository_id), None)
        if found is None:
            raise KeyError(f"Unknown repository: {repository_id}")
        return found

    def instance(self, instance_id: str) -> Instance:
        found = next((item for item in self.instances if item.id == instance_id), None)
        if found is None:
            raise KeyError(f"Unknown instance: {instance_id}")
        return found


class ConfigStore:
    """Loads and saves the configuration file."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lock = RLock()

    def load(self) -> Config:
        with self._lock:
            raw = json.loads(self.path.read_text())
        root = raw.get("models_root") or _root_of(raw.get("repositories", []))
        return Config(
            title=raw.get("title", "AI-Lab"),
            host=raw.get("host", "0.0.0.0"),
            port=int(raw.get("port", 8090)),
            models_root=root,
            repositories=[_under(root, item) for item in raw.get("repositories", [])],
            # `name` is dropped rather than rejected: a configuration written
            # before the label was removed still loads, and loses only the
            # label.
            instances=[Instance(**{key: value for key, value in item.items()
                                   if key != "name"})
                       for item in raw.get("instances", [])],
            engines=raw.get("engines", {}),
            gateway=raw.get("gateway", {}),
            memory=raw.get("memory", {}),
        )

    def save(self, config: Config) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_name(self.path.name + ".tmp")
            temporary.write_text(json.dumps(asdict(config), indent=2) + "\n")
            temporary.replace(self.path)

    @contextmanager
    def mutate(self) -> Iterator[Config]:
        """Load, hand over for editing, then save. Held under a lock throughout.

        Nothing is written if the block raises, so a rejected change leaves the
        file untouched.
        """
        with self._lock:
            config = self.load()
            yield config
            self.save(config)


def _under(root: str, stored: dict) -> Repository:
    """A repository, pointed at its folder under the models root.

    Whatever path was stored is ignored: there is one root now, and the rest
    follows from it. A configuration written before that still loads, and comes
    out pointing at the same place — see `_root_of`.
    """
    fields = {key: value for key, value in stored.items() if key != "path"}
    repository = Repository(**fields)
    return replace(repository,
                   path=str(Path(root) / repository.format) if root else "")


def _root_of(stored: list) -> str:
    """The models root of a configuration written before there was one.

    The directory every repository sat in. Read from what is there rather than
    guessed at, so a machine keeps pointing where it pointed: on the container
    that is `/models`, on the Mac `/Volumes/Marian_Backup/models`.

    Nothing when they did not share one — which cannot be expressed under one
    root, and is better left empty and visible than silently moved.
    """
    parents = {str(Path(item["path"]).parent)
               for item in stored if item.get("path")}
    return parents.pop() if len(parents) == 1 else ""
