"""Asking an engine whether it is ready.

Shared because every engine answers an HTTP endpoint, and because the answer
must never raise: during a load the process is not listening yet, and a
refused connection simply means "not yet".
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request


def http_ok(port: int, path: str = "/health", timeout: float = 1.0) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=timeout) as response:
            if response.status != 200:
                return False
            body = response.read()
    except (OSError, urllib.error.URLError, ValueError):
        return False
    try:
        payload = json.loads(body)
    except ValueError:
        return True          # a 200 with a non-JSON body still means it answered
    status = payload.get("status")
    return status in (None, "ok")
