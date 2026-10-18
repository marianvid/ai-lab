"""Public named image workflows and their AI-Lab-owned jobs."""

from __future__ import annotations

from typing import Any

from ...types import Task
from .jobs import register_jobs


def register(router, jobs: Any) -> None:
    """The image profiles, the image jobs, and the two OpenAI-shaped image requests.

    `jobs` is the image job queue (`media.image_jobs.ImageJobs`), built before
    the routes and handed in.
    """
    router.add("GET", "/api/image-profiles", lambda **_: jobs.profiles())
    register_jobs(router, "/api/image-jobs", jobs)
    router.add("POST", "/v1/images/generations",
               lambda body, **_: jobs.submit(body, Task.IMAGE_GENERATION))
    router.add("POST", "/v1/images/edits",
               lambda body, **_: jobs.submit(body, Task.IMAGE_EDIT))
