"""Routes for reclaimable files outside the model library."""


def register(router, operations) -> None:
    router.add("GET", "/api/storage", lambda **_: operations.storage_view())
    router.add("DELETE", "/api/storage/{id}",
               lambda id, **_: operations.clear_storage(id))
