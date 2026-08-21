"""The OpenAI-compatible front door.

One address for every configured model. A client names the model it wants; if
that model is not loaded it is loaded first, and the client only notices that
the first request took longer.

This file decides nothing about models. It reads the name out of the body, asks
`Gateway` for a lease, and forwards the request to whichever port that lease
points at.
"""

from __future__ import annotations

from ...gateway import Gateway
from ..passthrough import forward


def register(router, operations, gateway: Gateway) -> None:
    router.add("GET", "/v1/models", lambda **_: _catalogue(gateway))
    router.add("GET", "/api/gateway", lambda **_: gateway.stats())
    for path in ("/v1/chat/completions", "/v1/completions", "/v1/embeddings"):
        router.add("POST", path,
                   _forwarder(gateway, path))


def _catalogue(gateway: Gateway) -> dict:
    """Every configured model, in the shape an OpenAI client expects.

    Models that are not loaded are listed too. That is the point: a client is
    supposed to be able to ask for one of them.
    """
    data = []
    for row in gateway.catalogue():
        data.append({
            "id": row["id"],
            "object": "model",
            "owned_by": row["engine"],
            # Not part of the OpenAI shape, and harmless to a client that
            # ignores unknown fields. A person reading this by hand wants to
            # know which of these are up.
            "ai_lab": {"loaded": row["loaded"], "ready": row["ready"],
                       "port": row["port"], "aliases": row["aliases"]},
        })
    return {"object": "list", "data": data}


def _forwarder(gateway: Gateway, path: str):
    def handle(body=None, **_):
        payload = body or {}
        wanted = payload.get("model")
        if not wanted:
            raise ValueError("the request must name a model")

        # The lease is held until the last byte of the answer has been read, so
        # a swap cannot pull the model out from under a stream in progress.
        lease = gateway.acquire(wanted)

        # The engine knows its own model by a different name than the entry
        # does, and rejects a name it does not recognise. Ask by the name it
        # reports rather than passing ours through. The lease carries it, so
        # this costs nothing.
        outgoing = dict(payload)
        outgoing["model"] = lease.model_name or wanted

        url = f"http://127.0.0.1:{lease.port}{path}"
        try:
            return forward(url, outgoing, on_close=gateway.release)
        except Exception:
            gateway.release()
            raise
    return handle

