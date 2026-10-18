"""The routes every job queue has: list the jobs, read one, cancel one.

Image jobs and media jobs are queues of the same shape behind different paths.
The queue is looked up on each request (`queue()`), because the job queues are
attached to the operations after the routes are registered.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def register_jobs(router, path: str, queue: Callable[[], Any]) -> None:
    """`GET path`, `GET path/{id}` and `DELETE path/{id}` for the queue `queue()` returns."""
    router.add("GET", path, lambda **_: queue().list())
    router.add("GET", path + "/{id}",
               lambda id, **_: queue().get(id))
    router.add("DELETE", path + "/{id}",
               lambda id, **_: queue().cancel(id))
