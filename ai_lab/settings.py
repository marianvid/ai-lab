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

from . import budget
from pathlib import Path

from .config import ConfigStore
from .engines.registry import Registry
from .hosts.base import Host


class Settings:
    def __init__(self, store: ConfigStore, host: Host, engines: Registry) -> None:
        self.store = store
        self.host = host
        self.engines = engines

    def view(self) -> dict:
        config = self.store.load()
        capabilities = self.host.capabilities()
        return {
            "title": config.title,
            "host": {
                "operating_system": capabilities.operating_system,
                "supervisor": capabilities.supervisor,
                "accelerator_kind": capabilities.accelerator_kind,
                "can_configure_accelerator": capabilities.can_configure_accelerator,
            },
            "accelerator": asdict(self.host.accelerator()),
            # What is actually available for models, pool by pool. The page
            # reads this rather than doing the arithmetic itself, so the number
            # on screen is the same one an admission decision would use.
            "memory": budget.of(self.host, self.reserve_mb(config)).json(),
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
    def _repository(repository) -> dict:
        """A repository plus the free space on the disk holding it.

        Free space is reported per repository rather than per disk because
        that is the question being asked: can this model be downloaded here.

        `writable` is tested rather than trusted. The configuration declares an
        intention, but the directory belongs to the filesystem, and a download
        that is offered and then fails on a permission error is worse than one
        that was never offered.
        """
        path = Path(repository.path)
        exists = path.is_dir()
        row = {**asdict(repository), "exists": exists,
               "free_bytes": 0, "total_bytes": 0,
               "writable": bool(repository.writable and exists
                                and os.access(path, os.W_OK | os.X_OK))}
        if exists:
            usage = shutil.disk_usage(path)
            row["free_bytes"] = usage.free
            row["total_bytes"] = usage.total
        return row
