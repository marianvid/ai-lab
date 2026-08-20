"""One module per group of URLs.

Each module exposes `register(router, operations)` and nothing else. The
handlers translate a request into one call and return the result; anything
longer than a couple of lines belongs in `operations.py` instead.

`gateway` is the exception: it also needs the object that decides which entry
serves a model name, so it takes one more argument.
"""

from . import catalog, downloads, gateway, huggingface, runtime, settings

MODULES = (settings, catalog, runtime, downloads, huggingface)


def register_all(router, operations, model_gateway=None) -> None:
    for module in MODULES:
        module.register(router, operations)
    if model_gateway is not None:
        gateway.register(router, operations, model_gateway)
