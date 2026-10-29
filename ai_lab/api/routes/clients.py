"""Access: "is it up", the registered clients, the heartbeat, login, Security.

One call per route into `clients.access.Access`; nothing is decided here.
"""

from __future__ import annotations

from ..router import Reply


def register(router, access, journal=None) -> None:
    """The access routes, bound to `access` (`clients.access.Access`).

    `journal` is the operation log (`runtime.journal.OperationJournal`), read
    by the page.
    """
    router.add("GET", "/api/health", lambda **_: {"ok": True, "on": access.power.on})
    router.add("GET", "/api/power", lambda **_: access.power.view())
    router.add("POST", "/api/power", lambda body, **_: access.switch(bool(body.get("on"))))
    if journal is not None:
        router.add("GET", "/api/journal",
                   lambda query, **_: journal.since(int(query.get("after") or 0)))
    _clients(router, access)
    _web(router, access)


def _clients(router, access) -> None:
    def heartbeat(headers, **_):
        return {"received_at": access.heartbeat(headers), "on": access.power.on}

    router.add("GET", "/api/clients", lambda **_: access.clients())
    router.add("POST", "/api/clients",
               lambda body, **_: access.register(str(body.get("name", ""))))
    router.add("POST", "/api/clients/heartbeat", heartbeat)
    router.add("DELETE", "/api/clients/{name}",
               lambda name, **_: access.remove(name))


def _web(router, access) -> None:
    def login(body, **_):
        cookie = access.login(str(body.get("password", "")))
        return Reply({"ok": True}, {"Set-Cookie": cookie})

    def logout(headers, **_):
        return Reply({"ok": True}, {"Set-Cookie": access.logout(headers)})

    def password(body, **_):
        return access.set_password(str(body.get("new", "")), body.get("current"))

    router.add("GET", "/api/session", lambda headers, **_: access.session(headers))
    router.add("POST", "/api/login", login)
    router.add("POST", "/api/logout", logout)
    router.add("GET", "/api/security", lambda **_: access.security.view())
    router.add("PATCH", "/api/security",
               lambda body, **_: access.update_security(dict(body)))
    router.add("POST", "/api/security/password", password)
