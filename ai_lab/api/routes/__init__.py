"""One module per group of URLs.

Each module exposes `register(router, operations)` and nothing else. The
handlers translate a request into one call and return the result; anything
longer than a couple of lines belongs in `operations.py` instead. `images`
takes the image profiles instead.

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
from . import runtime
from . import settings
from . import storage

MODULES = (settings, storage, catalog, downloads, huggingface)
NO_LANES = gateway.NO_LANES


@dataclass(frozen=True)
class RouteParts:
    """What the routes are bound to besides the operations; each part optional.

    `gateway` decides which entry serves a model name; `lanes` are the lanes
    beside the card; `images` answers the named image profiles; `gpu_health` is the card's health monitor (or
    the reason there is none); `access` is who may call (tokens, login,
    Security, On/Off); `journal` is the operation log. A router built for a test leaves out what it does not need, and
    those routes are simply not there.
    """

    gateway: Any = None
    access: Any = None
    journal: Any = None
    lanes: Any = NO_LANES
    images: Any = None
    gpu_health: Any = None


def register_all(router, operations, parts: RouteParts | None = None) -> None:
    """Every module's routes, and those of each part that is there."""
    parts = parts or RouteParts()
    for module in MODULES:
        module.register(router, operations)
    runtime.register(router, operations, parts.gateway)
    if parts.gateway is not None:
        gateway.register(router, operations, parts.gateway, parts.lanes)
    if parts.images is not None:
        images.register(router, parts.images)
    if parts.gpu_health is not None:
        gpu_health.register(router, parts.gpu_health)
    if parts.access is not None:
        clients.register(router, parts.access, parts.journal)
