"""The progress stream.

A plain HTTP response that stays open and writes `data: {...}` lines as they
happen. Chosen over WebSocket because the traffic only goes one way and this
needs nothing beyond the standard library.

Each browser tab gets its own subscription with its own bounded queue, so a
tab left open on a sleeping laptop cannot slow a model load.
"""

from __future__ import annotations

import json

from ..events import EventBus, to_json

KEEPALIVE = b": keep-alive\n\n"


def stream(bus: EventBus, write, is_open) -> None:
    """Write events until the client goes away.

    `write` is expected to raise when the socket closes; that is the normal way
    this ends, so a broken pipe is not an error worth reporting.
    """
    subscription = bus.subscribe()
    try:
        for event in subscription.events():
            if not is_open():
                return
            if event is None:
                write(KEEPALIVE)
                continue
            payload = json.dumps(to_json(event)).encode()
            write(b"data: " + payload + b"\n\n")
    except (BrokenPipeError, ConnectionResetError, OSError):
        return
    finally:
        subscription.close()
