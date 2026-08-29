"""Routes for the library: what is on disk, and removing it."""

from __future__ import annotations


def register(router, operations) -> None:
    router.add("GET", "/api/models",
               lambda query=None, **_: operations.models((query or {}).get("engine")))
    router.add("GET", "/api/formats", lambda **_: operations.supported_formats())
    router.add("DELETE", "/api/models/{model_id}",
               lambda model_id, **_: operations.delete_model(model_id))
    router.add("POST", "/api/models/{model_id}/move",
               lambda model_id, body=None, **_: operations.move_model(
                   model_id, (body or {})["storage_tier"]))
    router.add("GET", "/api/models/moves", lambda **_: operations.move_jobs())
    router.add("POST", "/api/models/moves/{job_id}/cancel",
               lambda job_id, **_: operations.cancel_move(job_id))
