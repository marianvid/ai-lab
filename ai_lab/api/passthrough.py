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

# How long to wait for an engine. Both are limits of safety rather than of
# patience: in normal work nothing comes near them. The values are the
# gateway's, because they belong to the machine — a card that reads 8,400
# tokens of prompt in under a second and a Mac running a 70 GB model at 17
# tokens a second want very different numbers.
#
# There used to be one, of an hour, and it meant two different things depending
# on whether the client asked for streaming. Without streaming the engine sends
# nothing until it has finished, so the limit covered the whole answer; with
# streaming it covered the gap between chunks. One number for two jobs whose
# sane values are four orders of magnitude apart.
FIRST_BYTE_S = 120.0
BETWEEN_BYTES_S = 30.0


@dataclass(slots=True)
class Passthrough:
    status: int
    content_type: str
    chunks: Iterator[bytes]
    headers: dict[str, str]


def forward(url: str, payload: dict, on_close=None,
            first_byte_s: float = FIRST_BYTE_S,
            between_bytes_s: float = BETWEEN_BYTES_S) -> Passthrough:
    """POST a JSON body to an engine and hand back its answer as it arrives.

    `on_close` runs when the body has been fully read or the connection has
    broken. That is where the caller releases its lease — it must happen after
    the last byte, not when this function returns, because the answer is still
    being read then.

    The two limits are separate because they catch different faults. Nothing at
    all means the engine never started answering; a gap in the middle means it
    started and stopped. At the slowest generation measured on either machine,
    17 tokens a second, the gap between them is 59 milliseconds — so seconds of
    silence mid-answer is a fault, while a minute before the first byte can be
    an ordinary large prompt on a slow machine.
    """
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        response = urllib.request.urlopen(request, timeout=first_byte_s)
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

    # The answer has begun, so the tighter limit applies from here. Reached
    # through the socket underneath because that is where a read timeout lives;
    # if it cannot be reached, the first-byte limit keeps applying to every
    # read, which is looser than intended but still bounded.
    _tighten(response, between_bytes_s)

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


def _tighten(response, seconds: float) -> None:
    """Apply the between-bytes limit to the rest of this answer."""
    for owner in (getattr(response, "fp", None), response):
        raw = getattr(owner, "raw", owner)
        sock = getattr(raw, "_sock", None)
        if sock is not None:
            try:
                sock.settimeout(seconds)
                return
            except Exception:
                return
