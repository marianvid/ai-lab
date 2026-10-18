"""Browser and agent access to durable media jobs."""

from __future__ import annotations

from typing import Any

from .jobs import register_jobs


def register(router, jobs: Any) -> None:
    """The media jobs: list, read, cancel and submit.

    `jobs` is the media job queue (`media.jobs.MediaJobs`), built before the
    routes and handed in.
    """
    register_jobs(router, "/api/media-jobs", jobs)
    router.add("POST", "/api/media-jobs", lambda body, **_: jobs.submit(body))
