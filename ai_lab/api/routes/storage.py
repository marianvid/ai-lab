"""Routes for reclaimable files outside the model library."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...operations import StorageView


def register(router, operations: StorageView) -> None:
    """What can be reclaimed, and clearing it."""
    router.add("GET", "/api/storage", lambda **_: operations.storage_view())
    router.add("DELETE", "/api/storage/{id}",
               lambda id, **_: operations.clear_storage(id))
