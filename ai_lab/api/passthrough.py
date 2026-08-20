"""Forwarding one request to an engine and streaming the answer back.

The rest of the API answers with a dictionary and lets the server turn it into
JSON. This cannot: an engine's answer may be a stream that arrives over a minute
and has to reach the client as it appears, not after it finishes.

`Passthrough` is what a handler returns instead of a dictionary. It carries a
status, the headers worth keeping, and an iterator of byte chunks. The server
writes it out and still makes no decisions.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Iterator

# Headers worth copying from the engine's answer. Content-Length is deliberately
# not among them: the body is re-sent as it arrives, so the original length is
# either wrong or unknown.
KEEP = ("content-type", "cache-control")

CHUNK = 8192
TIMEOUT_S = 3600          # a long answer to a long prompt is still one request


@dataclass(slots=True)
class Passthrough:
    status: int
    content_type: str
    chunks: Iterator[bytes]
    headers: dict[str, str]


def forward(url: str, payload: dict, on_close=None) -> Passthrough:
    """POST a JSON body to an engine and hand back its answer as it arrives.

    `on_close` runs when the body has been fully read or the connection has
    broken. That is where the caller releases its lease — it must happen after
    the last byte, not when this function returns, because the answer is still
    being read then.
    """
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        response = urllib.request.urlopen(request, timeout=TIMEOUT_S)
    except urllib.error.HTTPError as error:
        # The engine refused it. Pass its own words through rather than
        # inventing a message: the client asked the engine a question and
        # deserves the engine's answer.
        body = error.read()
        if on_close:
            on_close()
        return Passthrough(
            status=error.code,
            content_type=error.headers.get("Content-Type", "application/json"),
            chunks=iter([body]),
            headers={})
    except Exception:
        if on_close:
            on_close()
        raise

    def read() -> Iterator[bytes]:
        try:
            while True:
                chunk = response.read(CHUNK)
                if not chunk:
                    return
                yield chunk
        finally:
            response.close()
            if on_close:
                on_close()

    headers = {name: value for name, value in response.headers.items()
               if name.lower() in KEEP}
    return Passthrough(
        status=response.status,
        content_type=response.headers.get("Content-Type", "application/json"),
        chunks=read(),
        headers=headers)
