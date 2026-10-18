"""Public named image workflows and their AI-Lab-owned jobs."""

from __future__ import annotations

from ...types import Task
from .jobs import register_jobs


def register(router, operations) -> None:
    """The image profiles, the image jobs, and the two OpenAI-shaped image requests."""
    router.add("GET", "/api/image-profiles",
               lambda **_: operations.image_jobs.profiles())
    register_jobs(router, "/api/image-jobs", lambda: operations.image_jobs)
    router.add("POST", "/v1/images/generations",
               lambda body, **_: operations.image_jobs.submit(
                   body, Task.IMAGE_GENERATION))
    router.add("POST", "/v1/images/edits",
               lambda body, **_: operations.image_jobs.submit(
                   body, Task.IMAGE_EDIT))
