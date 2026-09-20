"""Browser and agent access to durable media jobs."""


def register(router, operations) -> None:
    router.add("GET", "/api/media-jobs", lambda **_: operations.media_jobs.list())
    router.add("GET", "/api/media-jobs/{id}",
               lambda id, **_: operations.media_jobs.get(id))
    router.add("DELETE", "/api/media-jobs/{id}",
               lambda id, **_: operations.media_jobs.cancel(id))
    router.add("POST", "/api/media-jobs",
               lambda body, **_: operations.media_jobs.submit(body))
