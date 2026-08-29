"""Public named image workflows and their AI-Lab-owned jobs."""

from ...types import Task


def register(router, operations) -> None:
    router.add("GET", "/api/image-profiles",
               lambda **_: operations.image_jobs.profiles())
    router.add("GET", "/api/image-jobs",
               lambda **_: operations.image_jobs.list())
    router.add("GET", "/api/image-jobs/{id}",
               lambda id, **_: operations.image_jobs.get(id))
    router.add("DELETE", "/api/image-jobs/{id}",
               lambda id, **_: operations.image_jobs.cancel(id))
    router.add("POST", "/v1/images/generations",
               lambda body, **_: operations.image_jobs.submit(
                   body, Task.IMAGE_GENERATION))
    router.add("POST", "/v1/images/edits",
               lambda body, **_: operations.image_jobs.submit(
                   body, Task.IMAGE_EDIT))
