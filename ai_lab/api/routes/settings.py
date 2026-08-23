"""Routes for the settings screen: engine updates, and where models live."""

from __future__ import annotations


def register(router, operations) -> None:
    router.add("GET", "/api/settings", lambda **_: operations.settings_view())
    router.add("GET", "/api/builds", lambda **_: operations.build_status())
    # What an update would bring, read before anything is pressed. A GET
    # because it changes nothing: it reads a checkout, asks upstream what it
    # wrote, and asks the package manager what it would do.
    # Engines that arrive as packages. A new version is installed beside the
    # one in use, so these read and choose between folders rather than
    # replacing anything.
    # How much of this machine's memory is held back for the machine, and what
    # that leaves for models. A GET that reads the machine, not the file.
    router.add("GET", "/api/memory", lambda **_: operations.memory_budget())
    router.add("PATCH", "/api/memory",
               lambda body=None, **_: operations.update_memory(body or {}))
    router.add("GET", "/api/installs", lambda **_: operations.installs_available())
    router.add("GET", "/api/installs/{engine}",
               lambda engine, **_: operations.install_status(engine))
    router.add("POST", "/api/installs/{engine}",
               lambda engine, body=None, **_: operations.install_engine(
                   engine, (body or {}).get("version", "")))
    router.add("POST", "/api/installs/{engine}/{name}/activate",
               lambda engine, name, **_: operations.activate_install(engine, name))
    router.add("DELETE", "/api/installs/{engine}/{name}",
               lambda engine, name, **_: operations.remove_install(engine, name))
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
