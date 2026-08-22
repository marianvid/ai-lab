"""Routes for the settings screen: engine updates, and where models live."""

from __future__ import annotations


def register(router, operations) -> None:
    router.add("GET", "/api/settings", lambda **_: operations.settings_view())
    router.add("GET", "/api/builds", lambda **_: operations.build_status())
    # What an update would bring, read before anything is pressed. A GET
    # because it changes nothing: it reads a checkout, asks upstream what it
    # wrote, and asks the package manager what it would do.
    router.add("GET", "/api/builds/{engine}/changes",
               lambda engine, **_: operations.what_would_change(engine))
    router.add("POST", "/api/builds/{engine}/check",
               lambda engine, **_: operations.check_for_update(engine))
    router.add("POST", "/api/builds/{engine}/update",
               lambda engine, **_: operations.update_engine(engine))

    # Picking a folder: a web page cannot open a dialog on the server's
    # machine, so the server lists directories itself.
    router.add("GET", "/api/browse",
               lambda query=None, **_: operations.browse((query or {}).get("path")))
    router.add("POST", "/api/directories",
               lambda body=None, **_: operations.create_directory((body or {})["path"]))
    router.add("PATCH", "/api/repositories/{id}",
               lambda id, body=None, **_: operations.update_repository(id, body or {}))
