"""The routes every job queue has: list the jobs, read one, cancel one.

Image jobs and media jobs are queues of the same shape behind different paths.
The queue is built before the routes and handed in.
"""

from __future__ import annotations

from typing import Any


def register_jobs(router, path: str, queue: Any) -> None:
    """`GET path`, `GET path/{id}` and `DELETE path/{id}` for this job queue."""
    router.add("GET", path, lambda **_: queue.list())
    router.add("GET", path + "/{id}", lambda id, **_: queue.get(id))
    router.add("DELETE", path + "/{id}", lambda id, **_: queue.cancel(id))
