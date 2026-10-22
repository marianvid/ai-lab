"""Forwarding one request to an engine and streaming the answer back.

The rest of the API answers with a dictionary and lets the server turn it into
JSON. This cannot: an engine's answer may be a stream that arrives over a minute
and has to reach the client as it appears, not after it finishes.

`Passthrough` is what a handler returns instead of a dictionary. It carries a
status, the headers worth keeping, an iterator of byte chunks, and `close`,
which the server calls however the answer ends. The server writes it out and
still makes no decisions.

While it waits, the gateway keeps looking (`Waits`): a client that has left
gets its request stopped in the engine, and for an engine that reports its
progress (vLLM) a request is cut only when the engine as a whole has stopped
working — never because it waited (the owner's decision, 22 October 2026). An engine that
does not report keeps the two plain waits.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

from .upstream import END, FAILED, HEAD, Upstream

# Headers worth copying from the engine's answer. Content-Length is deliberately
# not among them: the body is re-sent as it arrives, so the original length is
# either wrong or unknown.
KEEP = ("content-type", "cache-control")

# How long to wait for an engine that does not report its progress. Both are
# limits of safety rather than of patience: in normal work nothing comes near
# them. The values are the gateway's, because they belong to the machine — a
# card that reads 8,400 tokens of prompt in under a second and a Mac running a
# 70 GB model at 17 tokens a second want very different numbers.
#
# There used to be one, of an hour, and it meant two different things depending
# on whether the client asked for streaming. Without streaming the engine sends
# nothing until it has finished, so the limit covered the whole answer; with
# streaming it covered the gap between chunks. One number for two jobs whose
# sane values are four orders of magnitude apart.
FIRST_BYTE_S = 120.0
BETWEEN_BYTES_S = 30.0
# For an engine that reports its progress (`gateway/limits.Patience`).
STALL_S = 60.0
SAFETY_S = 1800.0
# How often the gateway looks around while it waits.
TICK_S = 1.0
# A read on the engine's connection may outlast the gateway's own limits by
# this much before the socket itself gives up: the gateway decides first.
READ_MARGIN_S = 30.0


def _nothing() -> None:
    """The `close` of an answer that holds nothing."""


@dataclass(slots=True)
class Passthrough:
    """An engine's answer on its way to the client: status, headers and the body as it arrives.

    `close()` ends the answer and gives back what it holds — the engine's
    connection, the card place. The server calls it however writing ended,
    including when the client was gone before the first byte; it may be
    called more than once.
    """

    status: int
    content_type: str
    chunks: Iterator[bytes]
    headers: dict[str, str]
    close: Callable[[], None] = field(default=_nothing)


@dataclass(frozen=True, slots=True)
class Waits:
    """How long to wait for an engine, and what else to keep an eye on meanwhile.

    `first_byte_s` and `between_bytes_s`, for an engine that does not report
    its progress: the first byte of the answer, then the gaps between bytes.
    The two limits are separate because they catch different faults. Nothing
    at all means the engine never started answering; a gap in the middle means
    it started and stopped. At the slowest generation measured on either
    machine, 17 tokens a second, the gap between them is 59 milliseconds — so
    seconds of silence mid-answer is a fault, while a minute before the first
    byte can be an ordinary large prompt on a slow machine.

    `client_here()` turns False once the client has gone: its request is then
    stopped in the engine. `idle_for()` is how long the engine as a whole has
    done no work while it had some (`gateway/progress.py`); when given, it
    replaces the two plain waits: the request is cut after `stall_s` of that,
    or after `safety_s` in all, and never for waiting. `tick_s` is how often
    the gateway looks.
    """

    first_byte_s: float = FIRST_BYTE_S
    between_bytes_s: float = BETWEEN_BYTES_S
    client_here: Callable[[], bool] | None = None
    idle_for: Callable[[], float] | None = None
    stall_s: float = STALL_S
    safety_s: float = SAFETY_S
    tick_s: float = TICK_S


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
    # Told why, when the answer is cut because the engine stalled: the
    # gateway then restarts that engine (`Gateway.engine_stalled`).
    on_stall: Callable[[str], object] | None = None

    def closed(self) -> None:
        """Tell the caller the answer is over, if it asked."""
        if self.on_close:
            self.on_close()


class EngineStalledError(TimeoutError):
    """The engine had work and did none for the gateway's stall limit."""

    def __init__(self, idle_s: float) -> None:
        super().__init__(f"engine stalled: it did no work for {idle_s:.0f} s "
                         "while it had requests")
        self.explanation = str(self)


class ClientGoneError(ConnectionAbortedError):
    """The client left before its answer was ready; the request was stopped."""


# The defaults: the module's waits, and nobody to tell.
DEFAULT_WAITS = Waits()
NO_HOOKS = Hooks()


def forward(url: str, payload: dict | bytes, waits: Waits = DEFAULT_WAITS,
            hooks: Hooks = NO_HOOKS, content_type: str = "application/json") -> Passthrough:
    """POST a body to an engine and hand back its answer as it arrives.

    Returns once the answer's head has arrived — for a request that did not
    ask for streaming, that is when the engine has finished. An engine's
    refusal (a 4xx) comes back like any answer, with its own words: the client
    asked the engine a question and deserves the engine's answer.
    """
    data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    sent_at = time.perf_counter()
    read_limit_s = (waits.safety_s if waits.idle_for else waits.first_byte_s) + READ_MARGIN_S
    try:
        upstream = Upstream(url, data, content_type, read_limit_s)
    except Exception:
        hooks.closed()
        raise
    answer = _Answer(upstream, waits, hooks, sent_at)
    response = answer.head()
    headers = {name: value for name, value in response.headers.items()
               if name.lower() in KEEP}
    return Passthrough(
        status=response.status,
        content_type=response.headers.get("Content-Type", "application/json"),
        chunks=answer.chunks(),
        headers=headers,
        close=answer.close)


class _Answer:
    """One answer as the gateway waits for it, with the right limit at each moment.

    Headers are not the first byte, and neither is a keep-alive. A streaming
    engine sends its headers at once and then reads the prompt, which on a
    Mac takes a minute for a long one. During that time llama.cpp sends only
    an empty "still here" line (`:` on its own, an SSE comment) every 30 s.
    So the first-byte limit runs from when the request was sent until the
    first byte of real answer arrives; only then does the tighter idle limit
    take over. Keep-alives are still passed on to the client.
    """

    def __init__(self, upstream: Upstream, waits: Waits, hooks: Hooks, sent_at: float) -> None:
        self.upstream = upstream
        self.waits = waits
        self.hooks = hooks
        self.sent_at = sent_at
        self.began = False
        self.last_piece_at = sent_at
        self._closing = threading.Lock()
        self._closed = False

    def head(self):
        """The engine's response once its head has arrived; closes and raises on failure."""
        kind, value = self._next()
        if kind == HEAD:
            return value
        self.close()
        raise value if kind == FAILED else ConnectionError("the engine closed without answering")

    def chunks(self) -> Iterator[bytes]:
        """The answer's pieces as they arrive; closes the answer when done or broken."""
        try:
            while True:
                kind, value = self._next()
                if kind == END:
                    return
                if kind == FAILED:
                    raise value
                self._arrived(value)
                yield value
        finally:
            self.close()

    def close(self) -> None:
        """Stop reading, close the engine's connection and tell the caller — once."""
        with self._closing:
            if self._closed:
                return
            self._closed = True
        self.upstream.stop()
        self.hooks.closed()

    def _arrived(self, piece: bytes) -> None:
        """A piece of answer: the first real one starts the idle limit and is timed."""
        now = time.perf_counter()
        if self.began:
            self.last_piece_at = now
        elif not _keep_alive_only(piece):
            self.began = True
            self.last_piece_at = now
            if self.hooks.on_first_chunk:
                self.hooks.on_first_chunk(now - self.sent_at)

    def _next(self) -> tuple[str, Any]:
        """The next thing the engine sent, looking around between waits."""
        while True:
            try:
                return self.upstream.events.get(timeout=self._wait_s())
            except queue.Empty:
                pass
            reason = self._reason_to_stop(time.perf_counter())
            if reason is not None:
                self.close()
                if isinstance(reason, EngineStalledError) and self.hooks.on_stall:
                    self.hooks.on_stall(str(reason))
                raise reason

    def _wait_s(self) -> float:
        """How long to wait before looking around again."""
        if self.waits.idle_for:
            return self.waits.tick_s
        left = self._deadline() - time.perf_counter()
        return max(0.01, min(self.waits.tick_s, left))

    def _deadline(self) -> float:
        """When a plain wait runs out: the first byte, or the next one."""
        if self.began:
            return self.last_piece_at + self.waits.between_bytes_s
        return self.sent_at + self.waits.first_byte_s

    def _reason_to_stop(self, now: float) -> Exception | None:
        """Why this answer must end now, or None while it may go on."""
        if self.waits.client_here is not None and not self.waits.client_here():
            return ClientGoneError("the client left before its answer was ready")
        if self.waits.idle_for:
            idle_s = self.waits.idle_for()
            if idle_s >= self.waits.stall_s:
                return EngineStalledError(idle_s)
            if now - self.sent_at >= self.waits.safety_s:
                return TimeoutError("the gateway's safety limit was reached")
            return None
        if now >= self._deadline():
            return TimeoutError("the engine did not answer within the gateway's time limit")
        return None


def _keep_alive_only(chunk: bytes) -> bool:
    """Whether a piece of a stream is only "still here" and no answer.

    In the event-stream format a line starting with `:` is a comment, which
    engines send to keep a quiet connection open. Blank lines separate events.
    A piece made of nothing else carries no part of the answer.
    """
    return all(not line.strip() or line.startswith(b":")
               for line in chunk.split(b"\n"))
