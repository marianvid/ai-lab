"""Routes for the library: what is on disk, and removing it."""

from __future__ import annotations


def register(router, operations) -> None:
    router.add("GET", "/api/models",
               lambda query=None, **_: operations.models((query or {}).get("engine")))
    router.add("GET", "/api/formats", lambda **_: operations.supported_formats())
    router.add("DELETE", "/api/models/{model_id}",
               lambda model_id, **_: operations.delete_model(model_id))
