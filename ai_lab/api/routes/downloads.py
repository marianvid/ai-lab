"""Routes for transfers in progress."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...operations import DownloadsView


def register(router, operations: DownloadsView) -> None:
    """The transfers' routes: list, start, cancel."""
    router.add("GET", "/api/downloads", lambda **_: operations.transfers())
    router.add("POST", "/api/downloads",
               lambda body=None, **_: operations.download(
                   (body or {})["repo"], (body or {})["name"],
                   (body or {}).get("repository_id"),
                   (body or {}).get("storage_tier")))

    def cancel(id, **_):
        """Stop a download under way."""
        operations.cancel_download(id)
        return {"ok": True}

    router.add("POST", "/api/downloads/{id}/cancel", cancel)
