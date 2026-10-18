"""Browser and agent access to durable media jobs."""

from __future__ import annotations

from .jobs import register_jobs


def register(router, operations) -> None:
    """The media jobs: list, read, cancel and submit."""
    register_jobs(router, "/api/media-jobs", lambda: operations.media_jobs)
    router.add("POST", "/api/media-jobs",
               lambda body, **_: operations.media_jobs.submit(body))
