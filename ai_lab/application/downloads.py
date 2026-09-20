"""Catalog discovery and whole-model download destinations."""

from __future__ import annotations

import os
from pathlib import Path

from ..config import ConfigStore
from ..downloads import DownloadManager, HuggingFaceClient
from ..downloads import bundles
from ..hosts.base import Host
from ..engines.registry import Registry


class ModelDownloadService:
    def __init__(self, store: ConfigStore, downloads: DownloadManager,
                 huggingface: HuggingFaceClient, host: Host,
                 engines: Registry) -> None:
        self.store = store
        self.downloads = downloads
        self.huggingface = huggingface
        self.host = host
        self.engines = engines

    def supported_formats(self) -> list[str]:
        """Formats something on this machine can actually run.

        Used to filter what is offered for download: there is no point pulling
        30 GB of safetensors onto a machine with no engine that reads them.
        """
        capabilities = self.host.capabilities()
        formats: set[str] = set()
        for engine in self.engines.available(capabilities).values():
            formats.update(item.value for item in engine.formats())
        return sorted(formats)

    def search(self, query: str) -> dict:
        """Repositories holding something this machine can run.

        Only those. There was a switch to see the rest, and no answer to what
        it was for: a machine with no engine that reads safetensors cannot be
        helped by a list of them.

        `hidden` is what the filter took away, and it is the one thing the
        switch was good for. Nothing found and nothing *usable* found are
        different answers, and a list of length zero cannot tell them apart.
        """
        results = self.huggingface.search(query)
        supported = set(self.supported_formats())
        usable = [item for item in results
                  if supported.intersection(item["formats"])]
        return {"results": usable, "hidden": len(results) - len(usable)}

    def remote_sets(self, repo: str) -> list[dict]:
        """What a repository holds that this machine can run.

        Any bundle declared under this repository is listed too, first, because
        the individual parts below it cannot be used on their own.
        """
        supported = set(self.supported_formats())
        return [item.json()
                for item in self.huggingface.sets(repo, self._bundles())
                if item.format in supported]

    def _bundles(self):
        """The declared bundles, refused now if any of them is unsafe."""
        return bundles.parse(self.store.load().downloads.get("bundles", []))

    def download(self, repo: str, name: str,
                 repository_id: str | None = None,
                 storage_tier: str | None = None) -> dict:
        """Queue a complete model, into the repository that holds its format.

        The destination is worked out rather than asked for. A GGUF model
        belongs in the GGUF repository — the store is organised by format, and
        the listing already says which format this is, so making someone
        choose asks a question whose answer is already known. It can still be
        given explicitly when more than one repository holds a format.

        When it is given, it is checked first: a destination that cannot be
        written to should not cost a round trip to Hugging Face to discover.
        """
        config = self.store.load()
        selected_tier = storage_tier or config.download_root
        config.model_root(selected_tier)

        if repository_id:
            repository = config.repository(repository_id)
            if storage_tier and repository.root_id != storage_tier:
                raise ValueError(
                    f"{repository.name} is a {repository.root_id} repository; "
                    f"it cannot receive a {storage_tier} download. Choose a "
                    f"repository on the {storage_tier} tier, or drop the "
                    "explicit tier and let the repository decide.")
            destination = self._writable(repository)
            remote = self._remote_set(repo, name)
        else:
            remote = self._remote_set(repo, name)
            destination = self._repository_for(
                config, remote.format, selected_tier, remote.task)

        target = Path(destination.path) / Path(name).name
        return self.downloads.enqueue(
            remote, target, storage_tier=destination.root_id).json()

    def _remote_set(self, repo: str, name: str):
        remote = next((item for item in self.huggingface.sets(repo, self._bundles())
                       if item.name == name), None)
        if remote is None:
            raise KeyError(f"{name} is not in {repo}")
        return remote

    def _repository_for(self, config, format_name: str,
                        root_id: str = "core", task: str = ""):
        """The repository that holds this format and can be written to.

        One format can have more than one repository when the same engine does
        more than one job — ComfyUI generation and ComfyUI editing read the
        same kind of file and are kept apart. A set that says which job it is
        for picks the matching one instead of whichever comes first.
        """
        candidates = [item for item in config.repositories
                      if item.format == format_name
                      and item.root_id == root_id]
        if task:
            preferred = [item for item in candidates if item.task == task]
            candidates = preferred or candidates
        if not candidates:
            raise ValueError(
                f"No repository is configured for {format_name} models. "
                f"Add one in the configuration first.")
        errors = []
        for item in candidates:
            try:
                return self._writable_in_root(item, config.model_root(root_id))
            except ValueError as error:
                errors.append(str(error))
        raise ValueError(errors[0])

    @staticmethod
    def _writable(repository):
        """Return the repository, or explain why it cannot be written to.

        Writability is checked against the filesystem rather than the flag in
        the configuration: the flag records what was intended, and a download
        that starts and then dies on a permission error has wasted the wait.
        """
        path = Path(repository.path)
        if not repository.writable:
            raise ValueError(f"{repository.name} is marked read-only")
        if not path.is_dir():
            raise ValueError(f"{repository.name} does not exist at {path}")
        if not os.access(path, os.W_OK | os.X_OK):
            raise ValueError(
                f"{repository.name} is not writable by the manager. "
                f"Give it ownership of {path}.")
        return repository

    def _writable_in_root(self, repository, model_root):
        """Create a derived repository directory inside a valid model root.

        Format/task directories are consequences of the configured root, not
        operator-managed mount points. A first audio or image move should make
        its own subdirectory. The root itself is never created here: a missing
        mount must remain a visible error rather than silently writing to the
        container's underlying disk.
        """
        path = Path(repository.path)
        if path.is_dir():
            return self._writable(repository)
        root = Path(model_root.path)
        if not model_root.enabled:
            raise ValueError(f"{model_root.name} storage is disabled")
        if not model_root.writable:
            raise ValueError(f"{model_root.name} storage is marked read-only")
        if not root.is_dir():
            raise ValueError(
                f"{model_root.name} storage is not mounted at {root}. "
                "Connect or configure that storage, then try again.")
        resolved_root = root.resolve()
        candidate = path.resolve(strict=False)
        if not candidate.is_relative_to(resolved_root):
            raise ValueError(
                f"Refusing to create a model directory outside {resolved_root}")
        if not os.access(resolved_root, os.W_OK | os.X_OK):
            raise ValueError(
                f"{model_root.name} storage is not writable by the manager at "
                f"{resolved_root}")
        try:
            path.mkdir(parents=True)
        except OSError as error:
            raise ValueError(
                f"Could not prepare {model_root.name} storage for "
                f"{repository.name}: {error}") from None
        return self._writable(repository)

    def transfers(self) -> list[dict]:
        return self.downloads.list()

    def cancel_download(self, transfer_id: str) -> None:
        self.downloads.cancel(transfer_id)
