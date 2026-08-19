"""Routes for transfers in progress."""

from __future__ import annotations


def register(router, operations) -> None:
    router.add("GET", "/api/downloads", lambda **_: operations.transfers())
    router.add("POST", "/api/downloads",
               lambda body=None, **_: operations.download(
                   (body or {})["repo"], (body or {})["name"],
                   (body or {}).get("repository_id")))
    router.add("POST", "/api/downloads/{id}/cancel",
               lambda id, **_: operations.cancel_download(id) or {"ok": True})
