"""Access: "is it up", the registered clients, the heartbeat, login, Security.

One call per route into `clients.access.Access`; nothing is decided here.
"""

from __future__ import annotations

from collections.abc import Callable

from ..router import Reply


def always_on() -> bool:
    """The answer when there is no On/Off switch (a router built for a test)."""
    return True


def register(router, access, is_on: Callable[[], bool] = always_on) -> None:
    """The access routes, bound to `access` and to the On/Off switch."""
    router.add("GET", "/api/health", lambda **_: {"ok": True, "on": is_on()})
    _clients(router, access, is_on)
    _web(router, access)


def _clients(router, access, is_on: Callable[[], bool]) -> None:
    def heartbeat(headers, body, **_):
        return {"received_at": access.heartbeat(headers, body), "on": is_on()}

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
