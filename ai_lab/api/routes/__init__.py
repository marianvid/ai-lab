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

from ...lanes import SideLanes
from . import (
    catalog,
    downloads,
    gateway,
    huggingface,
    images,
    media,
    reservations,
    runtime,
    settings,
    storage,
)

MODULES = (settings, storage, catalog, downloads, huggingface, images, media)


def register_all(router, operations, model_gateway=None,
                 lanes: SideLanes = gateway.NO_LANES, book=None) -> None:
    """Every module's routes; the gateway's only when there is a gateway.

    `book` is the card's reservations (ADR 0005): the front door asks it before
    any model call, the Load buttons refuse while the card is reserved, and its
    own routes manage it. Without one, nothing is reserved.
    """
    for module in MODULES:
        module.register(router, operations)
    runtime.register(router, operations, model_gateway, book)
    if model_gateway is not None:
        gateway.register(router, operations, model_gateway, lanes, book)
    if book is not None:
        reservations.register(router, operations, book)
