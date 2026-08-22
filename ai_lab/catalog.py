"""What models are on disk.

Walks the configured repositories and groups loose files into complete models.
This is where the shard rule lives: the knowledge that
`model-00003-of-00005.safetensors` is one fifth of one thing, and that the
tokenizer sitting next to it belongs with it.

That rule used to live in the HTTP handler, which meant supporting a new
format required editing the web server. Here it can be tested against a
directory of empty files, with no HTTP and no real weights.

A model missing one of its shards is reported as incomplete rather than
hidden, so an interrupted download is visible instead of mysterious.
"""

from __future__ import annotations

from pathlib import Path

from .config import Repository
from .naming import base_name, is_companion, is_part, is_weight, missing_shards
from .types import Format, ModelFile, ModelSet


class Catalog:
    """Scans repositories and returns the models found in them.

    `known` remembers what each model's files said it can do, so they are
    opened once. Without one, nothing is worked out and every model reports no
    capabilities — which is what a test that does not care about them wants.
    """

    def __init__(self, known=None) -> None:
        self.known = known

    def scan(self, repositories: list[Repository]) -> list[ModelSet]:
        models: list[ModelSet] = []
        for repository in repositories:
            root = Path(repository.path)
            if not root.is_dir():
                continue
            models.extend(self._scan_repository(repository, root))
        models.sort(key=lambda item: item.id)
        return models

    def find(self, repositories: list[Repository], model_id: str) -> ModelSet:
        found = next((item for item in self.scan(repositories) if item.id == model_id), None)
        if found is None:
            raise KeyError(f"Unknown model: {model_id}")
        return found

    # -- internals ---------------------------------------------------------

    def _scan_repository(self, repository: Repository, root: Path) -> list[ModelSet]:
        """Turn the files in a repository into models.

        Scanning directory by directory rather than file by file is what lets
        companions be attached: a tokenizer belongs to the model it sits beside.

        How a directory divides into models depends on the format, and the two
        rules are opposites:

        GGUF puts a whole model in one file, so one directory can hold several
        unrelated ones. Weights are grouped by shard base name, and the name of
        the model is the name of its file.

        Safetensors spreads one model across a directory — weights, index,
        tokenizer, config — so the directory *is* the model and its name is the
        directory's. Applying the GGUF rule here splits a model into pieces:
        `model_mtp.safetensors` beside `model.safetensors` is a
        multi-token-prediction head, not a second model, and on its own it will
        not load.
        """
        models: list[ModelSet] = []
        for directory in self._directories(root):
            weights, companions = self._classify(directory)
            if not weights:
                continue
            if self._directory_is_the_model(repository):
                models.append(self._build(repository, root, directory,
                                          self._directory_name(root, directory),
                                          weights, companions))
                continue
            for base, shards in self._group(weights).items():
                models.append(self._build(repository, root, directory, base, shards, companions))
        return models

    @staticmethod
    def _directory_is_the_model(repository: Repository) -> bool:
        """Whether one directory holds exactly one model.

        True for every format the engine is handed a directory for — the same
        distinction `_entrypoint` makes, kept in step with it.
        """
        return repository.format != Format.GGUF.value

    @staticmethod
    def _directory_name(root: Path, directory: Path) -> str:
        """The model's name when the directory is the model.

        Weight files in these formats are always called `model-00001-of-...`,
        so naming from the file would call every model "model". The directory
        carries the real name. A repository root holding loose weights has no
        directory of its own to be named after, so it falls back to its own name.
        """
        return directory.name if directory != root else root.name

    @staticmethod
    def _directories(root: Path) -> list[Path]:
        return [root, *(path for path in sorted(root.rglob("*")) if path.is_dir())]

    @staticmethod
    def _classify(directory: Path) -> tuple[list[Path], list[Path]]:
        weights, companions = [], []
        for path in sorted(directory.iterdir()):
            if not path.is_file():
                continue
            # A part is weights, but not a model: it goes with whatever else
            # is in the directory rather than becoming an entry of its own.
            # The order matters — a vision projector passes `is_weight`.
            if is_weight(path.name) and not is_part(path.name):
                weights.append(path)
            elif is_companion(path.name) or is_part(path.name):
                companions.append(path)
        return weights, companions

    @staticmethod
    def _group(weights: list[Path]) -> dict[str, list[Path]]:
        """Group weight files by the model they belong to.

        A directory can hold several unrelated GGUF files, so grouping is by
        shard base name rather than by directory alone.
        """
        groups: dict[str, list[Path]] = {}
        for path in weights:
            groups.setdefault(base_name(path.name), []).append(path)
        return groups

    def _build(self, repository: Repository, root: Path, directory: Path,
               base: str, shards: list[Path], companions: list[Path]) -> ModelSet:
        shards = sorted(shards)
        complete, missing = missing_shards([path.name for path in shards])
        files = tuple(
            ModelFile(path=str(path), size_bytes=path.stat().st_size)
            for path in (*shards, *companions)
        )
        relative = directory.relative_to(root)
        # When the directory is the model, `base` is already the directory's
        # name, so listing the path as well would repeat it: nvfp4/gemma/gemma.
        parts = ([repository.id, *relative.parts[:-1], base]
                 if self._directory_is_the_model(repository) and relative.parts
                 else [repository.id, *relative.parts, base])
        entrypoint = str(self._entrypoint(repository, directory, shards))
        return ModelSet(
            id="/".join(parts),
            name=base,
            format=Format(repository.format),
            entrypoint=entrypoint,
            files=files,
            complete=complete,
            missing=missing,
            capabilities=self._can_do(repository, entrypoint, companions),
        )

    def _can_do(self, repository: Repository, entrypoint: str,
                companions: list[Path]) -> frozenset[str]:
        if self.known is None:
            return frozenset()
        return self.known.of(entrypoint,
                             repository.format == Format.GGUF.value,
                             [path.name for path in companions])

    @staticmethod
    def _entrypoint(repository: Repository, directory: Path, shards: list[Path]) -> Path:
        """What gets handed to the engine.

        llama.cpp is given the first shard and finds the rest itself. Engines
        reading safetensors are given the directory, because they need the
        index and the tokenizer alongside the weights.
        """
        if repository.format == Format.GGUF.value:
            return shards[0]
        return directory
