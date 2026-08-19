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
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
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
    path: str
    format: str
    writable: bool = True


@dataclass(slots=True)
class Instance:
    """One configured engine slot. Not necessarily running."""

    id: str
    name: str
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
        return Config(
            title=raw.get("title", "AI-Lab"),
            host=raw.get("host", "0.0.0.0"),
            port=int(raw.get("port", 8090)),
            repositories=[Repository(**item) for item in raw.get("repositories", [])],
            instances=[Instance(**item) for item in raw.get("instances", [])],
            engines=raw.get("engines", {}),
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
