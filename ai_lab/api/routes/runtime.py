"""Routes for the model list: what is configured, and starting and stopping."""

from __future__ import annotations


def register(router, operations) -> None:
    router.add("GET", "/api/instances", lambda **_: operations.instances())
    router.add("GET", "/api/instances/new", lambda **_: operations.new_instance_form())
    router.add("POST", "/api/instances",
               lambda body=None, **_: operations.create_instance(body or {}))
    router.add("PATCH", "/api/instances/{id}",
               lambda id, body=None, **_: operations.update_instance(id, body or {}))
    router.add("DELETE", "/api/instances/{id}",
               lambda id, **_: operations.delete_instance(id) or {"ok": True})
    router.add("POST", "/api/instances/{id}/load",
               lambda id, **_: operations.load(id).json())
    router.add("POST", "/api/instances/{id}/unload",
               lambda id, **_: operations.unload(id).json())
    # Saving settings and restarting with them, as one action.
    router.add("POST", "/api/instances/{id}/apply",
               lambda id, body=None, **_: operations.apply_and_reload(id, body or {}))
