"""One module per group of URLs.

Each module exposes `register(router, operations)` and nothing else. The
handlers translate a request into one call and return the result; anything
longer than a couple of lines belongs in `operations.py` instead. The two job
queues' modules (`images`, `media`) take their queue instead.

Two modules take one more argument: the object that decides which entry serves
a model name. `gateway` needs it because that is its whole subject, and
`runtime` needs it to ask whether a model is mid-answer before stopping it.
Both work without it — a router built for a test has no gateway, and then
nothing is guarded because nothing is being served.
"""

from dataclasses import dataclass
from typing import Any

from . import catalog
from . import clients
from . import downloads
from . import gateway
from . import gpu_health
from . import huggingface
from . import images
from . import media
from . import reservations
from . import runtime
from . import settings
from . import storage

MODULES = (settings, storage, catalog, downloads, huggingface)
NO_LANES = gateway.NO_LANES


@dataclass(frozen=True)
class RouteParts:
    """What the routes are bound to besides the operations; each part optional.

    `gateway` decides which entry serves a model name; `lanes` are the lanes
    beside the card; `book` is the card's reservations (ADR 0005): the front
    door asks it before any model call, the Load buttons refuse while the card
    is reserved, and its own routes manage it, saving their settings in
    `store` (the configuration store); `image_jobs` and `media_jobs`
    are the two job queues; `gpu_health` is the card's health monitor (or
    the reason there is none); `access` is who may call (tokens, login,
    Security). A router built for a test leaves out what it does not need, and
    those routes are simply not there.
    """

    gateway: Any = None
    access: Any = None
    lanes: Any = NO_LANES
    book: Any = None
    store: Any = None
    image_jobs: Any = None
    media_jobs: Any = None
    gpu_health: Any = None


def register_all(router, operations, parts: RouteParts | None = None) -> None:
    """Every module's routes, and those of each part that is there."""
    parts = parts or RouteParts()
    for module in MODULES:
        module.register(router, operations)
    runtime.register(router, operations, parts.gateway, parts.book)
    if parts.gateway is not None:
        gateway.register(router, operations, parts.gateway, parts.lanes, parts.book)
    if parts.book is not None:
        reservations.register(router, parts.store, parts.book)
    if parts.image_jobs is not None:
        images.register(router, parts.image_jobs)
    if parts.media_jobs is not None:
        media.register(router, parts.media_jobs)
    if parts.gpu_health is not None:
        gpu_health.register(router, parts.gpu_health)
    if parts.access is not None:
        clients.register(router, parts.access)
