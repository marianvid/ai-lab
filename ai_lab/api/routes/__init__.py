"""One module per group of URLs.

Each module exposes `register(router, operations)` and nothing else. The
handlers translate a request into one call and return the result; anything
longer than a couple of lines belongs in `operations.py` instead.
"""

from . import catalog, downloads, huggingface, runtime, settings

MODULES = (settings, catalog, runtime, downloads, huggingface)


def register_all(router, operations) -> None:
    for module in MODULES:
        module.register(router, operations)
