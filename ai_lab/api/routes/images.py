"""The named image profiles and the two OpenAI-shaped image requests."""

from __future__ import annotations

from typing import Any

from ...types import Task


def register(router, images: Any) -> None:
    """The image profiles and image calls (`media.images.ImageCalls`)."""
    router.add("GET", "/api/image-profiles", lambda **_: images.profiles())
    router.add("POST", "/v1/images/generations",
               lambda body, **_: images.submit(body, Task.IMAGE_GENERATION))
    router.add("POST", "/v1/images/edits",
               lambda body, **_: images.submit(body, Task.IMAGE_EDIT))
