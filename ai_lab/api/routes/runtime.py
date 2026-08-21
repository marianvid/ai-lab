"""Routes for the model list: what is configured, and starting and stopping.

Starting and stopping are guarded. The gateway hands the card to one request at
a time, so agent traffic cannot interrupt itself — but these routes are the
buttons on the page, and they reach the engines directly. Without the guard,
pressing Unload while an answer is streaming kills it mid sentence, and the
agent on the other end sees a connection that simply stopped.

The guard refuses; it does not decide. A request carrying `force` goes ahead
anyway, because a model wedged in a bad state has to be stoppable. The page asks
first and sends `force` only if the person says so.
"""

from __future__ import annotations


def register(router, operations, gateway=None) -> None:
    def guard(action: str, instance_id: str, body: dict) -> dict:
        """Refuse to interrupt an answer in progress, and strip the override.

        `force` is an instruction to this layer, not a setting to be saved, so
        it never reaches `operations` — `apply` would otherwise store it beside
        the engine's own settings.
        """
        rest = {key: value for key, value in body.items() if key != "force"}
        if gateway is not None and not body.get("force"):
            gateway.guard(action, instance_id)
        return rest

    def load(id, body=None, **_):
        guard("load", id, body or {})
        return operations.load(id).json()

    def unload(id, body=None, **_):
        guard("unload", id, body or {})
        return operations.unload(id).json()

    def apply(id, body=None, **_):
        changes = guard("apply & reload", id, body or {})
        return operations.apply_and_reload(id, changes)

    router.add("GET", "/api/instances", lambda **_: operations.instances())
    router.add("GET", "/api/instances/new", lambda **_: operations.new_instance_form())
    router.add("POST", "/api/instances",
               lambda body=None, **_: operations.create_instance(body or {}))
    router.add("PATCH", "/api/instances/{id}",
               lambda id, body=None, **_: operations.update_instance(id, body or {}))
    # Deleting needs no guard: it already refuses while the entry is running,
    # so there is never an answer in progress to interrupt.
    router.add("DELETE", "/api/instances/{id}",
               lambda id, **_: operations.delete_instance(id) or {"ok": True})
    router.add("POST", "/api/instances/{id}/load", load)
    router.add("POST", "/api/instances/{id}/unload", unload)
    # Saving settings and restarting with them, as one action.
    router.add("POST", "/api/instances/{id}/apply", apply)
