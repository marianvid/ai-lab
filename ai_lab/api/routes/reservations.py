"""Card reservations (ADR 0005): a team asks for the card, keeps it alive, gives it back.

Each handler is one call into `ReservationBook`. The settings are stored in the
configuration's `reservations` section and used at once.
"""

from __future__ import annotations

from ...reservations import ReservationBook, ReservationPolicy


def register(router, operations, book: ReservationBook) -> None:
    """The reservation routes and the page's settings."""
    router.add("POST", "/api/reservations",
               lambda body=None, **_: book.request(body or {}))
    router.add("GET", "/api/reservations", lambda **_: book.overview())
    router.add("GET", "/api/reservations/{id}", lambda id, **_: book.get(id))
    router.add("POST", "/api/reservations/{id}/heartbeat",
               lambda id, **_: book.heartbeat(id))
    router.add("POST", "/api/reservations/{id}/extend",
               lambda id, body=None, **_: book.extend(id, (body or {}).get("seconds")))
    router.add("DELETE", "/api/reservations/{id}", lambda id, **_: book.release(id))
    router.add("POST", "/api/reservations/{id}/cancel", lambda id, **_: book.cancel(id))

    def settings(body=None, **_):
        """Change the reservation settings, keep them, and use them at once."""
        with operations.store.mutate() as config:
            policy = ReservationPolicy.from_mapping({**config.reservations, **(body or {})})
            config.reservations = policy.json()
        book.apply_policy(policy)
        return book.overview()

    router.add("PATCH", "/api/reservations/settings", settings)
