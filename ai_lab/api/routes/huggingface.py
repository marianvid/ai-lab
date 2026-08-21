"""Routes for browsing Hugging Face.

Only what this machine can run is listed, and there is no way to ask for the
rest. Pulling thirty gigabytes of a format no engine here reads is not a thing
anyone needed to do, and a switch offering it was one more decision in front of
a search box.

The search answer carries how many were filtered out, because "nothing found"
and "nothing you can run" are different answers and an empty list cannot tell
them apart.
"""

from __future__ import annotations


def register(router, operations) -> None:
    router.add("GET", "/api/hf/search",
               lambda query=None, **_: operations.search((query or {}).get("q", "")))
    router.add("GET", "/api/hf/sets",
               lambda query=None, **_: operations.remote_sets(
                   (query or {}).get("repo", "")))
