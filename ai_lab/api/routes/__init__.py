"""One module per group of URLs.

Each module exposes `register(router, operations)` and nothing else. The
handlers translate a request into one call and return the result; anything
longer than a couple of lines belongs in `operations.py` instead.

Two modules take one more argument: the object that decides which entry serves
a model name. `gateway` needs it because that is its whole subject, and
`runtime` needs it to ask whether a model is mid-answer before stopping it.
Both work without it — a router built for a test has no gateway, and then
nothing is guarded because nothing is being served.
"""

from . import catalog, downloads, gateway, huggingface, runtime, settings

MODULES = (settings, catalog, downloads, huggingface)


def register_all(router, operations, model_gateway=None) -> None:
    for module in MODULES:
        module.register(router, operations)
    runtime.register(router, operations, model_gateway)
    if model_gateway is not None:
        gateway.register(router, operations, model_gateway)
