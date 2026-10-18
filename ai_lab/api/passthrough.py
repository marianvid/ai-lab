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
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from dataclasses import dataclass

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
    """An engine's answer on its way to the client: status, headers and the body as it arrives."""

    status: int
    content_type: str
    chunks: Iterator[bytes]
    headers: dict[str, str]


@dataclass(frozen=True, slots=True)
class Waits:
    """How long to wait for the first byte of the answer, and then between bytes.

    The two limits are separate because they catch different faults. Nothing
    at all means the engine never started answering; a gap in the middle
    means it started and stopped. At the slowest generation measured on either
    machine, 17 tokens a second, the gap between them is 59 milliseconds — so
    seconds of silence mid-answer is a fault, while a minute before the first
    byte can be an ordinary large prompt on a slow machine.
    """

    first_byte_s: float = FIRST_BYTE_S
    between_bytes_s: float = BETWEEN_BYTES_S


@dataclass(frozen=True, slots=True)
class Hooks:
    """What the caller wants told: the body is done, and how long the first byte took.

    `on_close` runs when the body has been fully read or the connection has
    broken. That is where the caller releases its lease — it must happen after
    the last byte, not when `forward` returns, because the answer is still
    being read then. `on_first_chunk` gets the seconds from sending to the
    first byte of real answer.
    """

    on_close: Callable[[], None] | None = None
    on_first_chunk: Callable[[float], None] | None = None

    def closed(self) -> None:
        """Tell the caller the answer is over, if it asked."""
        if self.on_close:
            self.on_close()


# The defaults: the module's waits, and nobody to tell.
DEFAULT_WAITS = Waits()
NO_HOOKS = Hooks()


def forward(url: str, payload: dict | bytes, waits: Waits = DEFAULT_WAITS,
            hooks: Hooks = NO_HOOKS, content_type: str = "application/json") -> Passthrough:
    """POST a body to an engine and hand back its answer as it arrives."""
    data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": content_type}, method="POST")
    sent_at = time.perf_counter()
    try:
        response = urllib.request.urlopen(request, timeout=waits.first_byte_s)
    except urllib.error.HTTPError as error:
        # The engine refused it. Pass its own words through rather than
        # inventing a message: the client asked the engine a question and
        # deserves the engine's answer.
        body = error.read()
        hooks.closed()
        return Passthrough(
            status=error.code,
            content_type=error.headers.get("Content-Type", "application/json"),
            chunks=iter([body]),
            headers={})
    except Exception:
        hooks.closed()
        raise
    reader = _Reader(response, waits, hooks, sent_at)
    headers = {name: value for name, value in response.headers.items()
               if name.lower() in KEEP}
    return Passthrough(
        status=response.status,
        content_type=response.headers.get("Content-Type", "application/json"),
        chunks=reader.chunks(),
        headers=headers)


class _Reader:
    """Reads one answer piece by piece, with the right limit at each moment.

    Headers are not the first byte, and neither is a keep-alive. A streaming
    engine sends its headers at once and then reads the prompt, which on a
    Mac takes a minute for a long one. During that time llama.cpp sends only
    an empty "still here" line (`:` on its own, an SSE comment) every 30 s.
    Starting the idle limit at the headers cut those answers off at 30 s, just
    before the first keep-alive, and the client got an empty stream. So the
    first-byte limit runs from when the request was sent until the first byte
    of real answer arrives; only then does the tighter idle limit take over.
    Keep-alives are still passed on to the client.
    """

    def __init__(self, response, waits: Waits, hooks: Hooks, sent_at: float) -> None:
        self.response = response
        self.waits = waits
        self.hooks = hooks
        self.sent_at = sent_at
        # `read1` hands over whatever has arrived. Plain `read(CHUNK)` waits
        # until it has CHUNK bytes or the answer ends, so a stream reached the
        # client in 8 KB bursts — dozens of words late.
        self.take = getattr(response, "read1", None) or response.read
        _limit(response, self._remaining())

    def _remaining(self) -> float:
        return max(0.1, self.waits.first_byte_s - (time.perf_counter() - self.sent_at))

    def chunks(self) -> Iterator[bytes]:
        """The answer's pieces as they arrive; closes the answer when done or broken."""
        first = True
        try:
            while True:
                chunk = self.take(CHUNK)
                if not chunk:
                    return
                if first and _keep_alive_only(chunk):
                    # Not an answer yet: the first-byte limit keeps running,
                    # counted from when the request was sent.
                    _limit(self.response, self._remaining())
                elif first:
                    first = False
                    self._answer_began()
                yield chunk
        finally:
            self.response.close()
            self.hooks.closed()

    def _answer_began(self) -> None:
        """The tighter limit from here; and tell the caller how long the engine took."""
        _limit(self.response, self.waits.between_bytes_s)
        if self.hooks.on_first_chunk:
            # Timed here because this is the only place that sees the moment
            # it arrives. It looks at a piece only to tell a keep-alive from
            # an answer, never further.
            self.hooks.on_first_chunk(time.perf_counter() - self.sent_at)


def _keep_alive_only(chunk: bytes) -> bool:
    """Whether a piece of a stream is only "still here" and no answer.

    In the event-stream format a line starting with `:` is a comment, which
    engines send to keep a quiet connection open. Blank lines separate events.
    A piece made of nothing else carries no part of the answer.
    """
    return all(not line.strip() or line.startswith(b":")
               for line in chunk.split(b"\n"))


def _limit(response, seconds: float) -> None:
    """Set how long each further read of this answer may wait.

    Reached through the socket underneath, because that is where a read
    timeout lives. If it cannot be reached, the first-byte limit given to
    `urlopen` keeps applying to every read: looser than intended, but bounded.
    """
    for owner in (getattr(response, "fp", None), response):
        raw = getattr(owner, "raw", owner)
        sock = getattr(raw, "_sock", None)
        if sock is not None:
            try:
                sock.settimeout(seconds)
                return
            except Exception:
                return
