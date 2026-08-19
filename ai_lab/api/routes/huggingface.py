"""Routes for browsing Hugging Face."""

from __future__ import annotations


def register(router, operations) -> None:
    router.add("GET", "/api/hf/search",
               lambda query=None, **_: operations.search(
                   (query or {}).get("q", ""),
                   only_supported=(query or {}).get("all") != "1"))
    # `all=1` lifts the filter, for pulling weights ahead of the engine that
    # will read them.
    router.add("GET", "/api/hf/sets",
               lambda query=None, **_: operations.remote_sets(
                   (query or {}).get("repo", ""),
                   only_supported=(query or {}).get("all") != "1"))
