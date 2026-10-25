"""The settings view: configuration plus the state of the machine.

Assembles the first screen — repositories with their free space, the
accelerator, and which engines are usable here. It reads from the config store
and the host and holds no state of its own.

Accelerator settings are read-only by decision rather than by omission: power
limits and similar are changed deliberately elsewhere, not through a web page.

Watch the naming, which is easy to trip over. `config.py` is the file store,
what gets written to config.json. This module is the view assembled from it.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import asdict
from pathlib import Path

from . import budget
from .config.store import ConfigStore
from .engines.registry import Registry
from .hosts.base import Host


class Settings:
    """The first settings screen, assembled from the configuration and the host."""

    def __init__(self, store: ConfigStore, host: Host, engines: Registry) -> None:
        self.store = store
        self.host = host
        self.engines = engines

    def view(self) -> dict:
        """Repositories with their free space, the accelerator, memory and engines."""
        config = self.store.load()
        capabilities = self.host.capabilities()
        # One reading, used twice. Each is an `nvidia-smi` on Linux — 30 ms —
        # and this page was asking for three to draw itself. It also stops the
        # two figures below disagreeing, which readings a moment apart do.
        card = self.host.accelerator()
        return {
            "title": config.title,
            "host": {
                "operating_system": capabilities.operating_system,
                "supervisor": capabilities.supervisor,
                "accelerator_kind": capabilities.accelerator_kind,
                "can_configure_accelerator": capabilities.can_configure_accelerator,
            },
            "accelerator": asdict(card),
            # What is actually available for models, pool by pool. The page
            # reads this rather than doing the arithmetic itself, so the number
            # on screen is the same one an admission decision would use.
            "memory": budget.of(self.host, self.reserve_mb(config), card=card,
                                ceiling_mb=self.ceiling_mb(config)).json(),
            "models_root": config.models_root,
            "model_roots": [self._model_root(item) for item in config.model_roots],
            "download_root": config.download_root,
            "repositories": [self._repository(item) for item in config.repositories],
            "engines": self.engines.describe(capabilities),
        }

    @staticmethod
    def reserve_mb(config) -> float:
        """How much to leave for the machine itself."""
        try:
            value = float((config.memory or {}).get("reserve_mb",
                                                    budget.DEFAULT_RESERVE_MB))
        except (TypeError, ValueError):
            return budget.DEFAULT_RESERVE_MB
        return max(0.0, value)

    @staticmethod
    def ceiling_mb(config) -> float:
        """The most models may hold on unified memory (A12); 0: no ceiling."""
        try:
            value = float((config.memory or {}).get("ceiling_mb", 0.0))
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, value)

    @staticmethod
    def _repository(repository) -> dict:
        """A repository plus the free space on the disk holding it.

        Free space is reported per repository rather than per disk because
        that is the question being asked: can this model be downloaded here.

        `writable` is tested rather than trusted. The configuration declares an
        intention, but the directory belongs to the filesystem, and a download
        that is offered and then fails on a permission error is worse than one
        that was never offered.
        """
        # An empty path means the models root is not set, which is not the
        # same as a folder that is there. `Path("")` is the current working
        # directory and exists, so asking without checking would report a
        # configured store where there is none.
        path = Path(repository.path) if repository.path else None
        return {**asdict(repository), **_space(path, repository.writable)}

    @staticmethod
    def _model_root(model_root) -> dict:
        path = Path(model_root.path) if model_root.path else None
        return {**asdict(model_root),
                **_space(path if model_root.enabled else None, model_root.writable)}


def _space(path: Path | None, writable: bool) -> dict:
    """Whether a folder is there, how full its disk is, and whether it can be written.

    `writable` is the configuration's intention; it is tested against the
    filesystem rather than trusted.
    """
    if path is None or not path.is_dir():
        return {"exists": False, "free_bytes": 0, "total_bytes": 0, "writable": False}
    usage = shutil.disk_usage(path)
    return {"exists": True, "free_bytes": usage.free, "total_bytes": usage.total,
            "writable": bool(writable and os.access(path, os.W_OK | os.X_OK))}
