"""The front door: one address for every configured model.

A client names the model it wants; if that model is not loaded it is loaded
first, and the client only notices that the first request took longer.

Requests arrive in one of two shapes. Nearly everything speaks the OpenAI one.
A client written against Anthropic's own library speaks the other, and only
some engines answer it — vLLM does, llama.cpp does not. Both are accepted here
and an entry that cannot answer a shape is refused by name, because the
alternative is forwarding the request to an engine that replies 404 about a
path the client never chose.

This file decides nothing about models. It reads the name out of the body, asks
`Gateway` for a lease, and forwards the request to whichever port that lease
points at.
"""

from __future__ import annotations

from ...engines.base import ANTHROPIC_PATHS, OPENAI_PATHS
from ...gateway import Gateway
from ..passthrough import forward

# Every shape any engine here can answer. Registered as routes whatever is
# configured: a path that exists and explains why this model cannot serve it is
# more use than one that does not exist at all.
FORWARDED = OPENAI_PATHS + ANTHROPIC_PATHS


def register(router, operations, gateway: Gateway) -> None:
    router.add("GET", "/v1/models", lambda **_: _catalogue(gateway))
    router.add("GET", "/api/gateway", lambda **_: gateway.stats())

    def settings(body=None, **_):
        """Change the front door's own limits, and use them at once."""
        saved = operations.update_gateway(body or {})
        gateway.apply_settings(saved)
        return gateway.stats()

    router.add("PATCH", "/api/gateway", settings)
    for path in FORWARDED:
        router.add("POST", path, _forwarder(gateway, path))


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
                       "port": row["port"], "aliases": row["aliases"],
                       "shapes": row["shapes"]},
        })
    return {"object": "list", "data": data}


# Where a client puts settings the model has to be *started* with, rather than
# ones that go in a request. Anything else in the body belongs to the engine and
# is passed through untouched.
SETTINGS_FIELD = "ai_lab"


def _forwarder(gateway: Gateway, path: str):
    def handle(body=None, alive=None, **_):
        payload = body or {}
        wanted = payload.get("model")
        if not wanted:
            raise ValueError("the request must name a model")

        # Settings that decide how the model starts — context size and the
        # rest — cannot be part of the request the engine sees: the engine does
        # not know them, and would ignore them without a word. They travel in a
        # field of ours, which is read here and removed before forwarding, so
        # what reaches the engine is exactly what would have reached it before.
        settings = payload.get(SETTINGS_FIELD) or None
        if settings is not None and not isinstance(settings, dict):
            raise ValueError(f"{SETTINGS_FIELD} must be an object of settings")

        # The lease is held until the last byte of the answer has been read, so
        # a swap cannot pull the model out from under a stream in progress.
        # `path` goes with it: an entry whose engine does not answer this shape
        # is refused before anything is loaded, not after.
        lease = gateway.acquire(wanted, shape=path, settings=settings,
                                still_wanted=alive)

        # The engine knows its own model by a different name than the entry
        # does, and rejects a name it does not recognise. Ask by the name it
        # reports rather than passing ours through. The lease carries it, so
        # this costs nothing.
        outgoing = dict(payload)
        outgoing.pop(SETTINGS_FIELD, None)
        outgoing["model"] = lease.model_name or wanted

        url = f"http://127.0.0.1:{lease.port}{path}"
        try:
            return forward(url, outgoing, on_close=lease.release,
                           first_byte_s=gateway.first_byte_s,
                           between_bytes_s=gateway.between_bytes_s)
        except Exception:
            lease.release()
            raise
    return handle

