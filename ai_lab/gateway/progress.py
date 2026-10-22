"""Whether an engine is still getting work done, read from the engine's own counters.

A request that waits is not stuck. vLLM parks requests when its context pool
is full and serves them as others finish; from outside, a parked request and a
stuck one look the same — no answer yet. What tells them apart is the engine:
while it keeps processing tokens for somebody, everything waiting in it will
get its turn. Only an engine that has work and does none at all is stuck.

vLLM counts every token it processes, prompt and answer, step by step
(`tokens_processed` in its `live` report, from `vllm:iteration_tokens_total`).
`EngineProgress` reads that report at most every few seconds, however many
requests are asking, and says how long the engine has gone without any.

An engine that does not report (`live` empty) cannot be judged this way; the
gateway then keeps its two plain waits (`passthrough.Waits`).
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

# How often the engine's report is read, at most, whatever the number of
# requests watching it. Small against the stall limit (60 s by default).
READ_EVERY_S = 2.0
# More reads than this missed in a row: nobody was watching the engine.
UNWATCHED_READS = 3
WORK = "tokens_processed"


class EngineProgress:
    """One engine's progress, shared by every request to it.

    `read()` is the engine's `live` report (a dict; empty when it could not be
    read). Progress is any change in `tokens_processed`, or an engine with
    nothing to do (no request running or waiting): an idle engine is not
    stuck. A report that cannot be read counts as no progress, because an
    engine too stuck to answer its own counters is the case being looked for.
    """

    def __init__(self, read: Callable[[], dict], clock: Callable[[], float] = time.monotonic,
                 read_every_s: float = READ_EVERY_S) -> None:
        self._read = read
        self._clock = clock
        self._read_every_s = read_every_s
        self._lock = threading.Lock()
        self._last_read = float("-inf")
        self._last_work: float | None = None
        self._last_progress = clock()

    def idle_for(self) -> float:
        """Seconds since the engine last did any work while it had some."""
        with self._lock:
            now = self._clock()
            if now - self._last_read > self._read_every_s * UNWATCHED_READS:
                # Nobody was asking: what happened meanwhile is unknown, so
                # the clock starts now rather than at the last look, hours ago.
                self._last_progress = now
            if now - self._last_read >= self._read_every_s:
                self._last_read = now
                self._look(self._read(), now)
            return now - self._last_progress

    def _look(self, report: dict, now: float) -> None:
        work = report.get(WORK)
        if work is None:
            return
        idle = not report.get("requests_running") and not report.get("requests_waiting")
        if idle or work != self._last_work:
            self._last_progress = now
        self._last_work = work


class ProgressBook:
    """The `EngineProgress` of each engine port, made on first use."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._engines: dict[tuple[str, int], EngineProgress] = {}

    def of(self, instance_id: str, port: int, read: Callable[[], dict]) -> EngineProgress:
        """The shared watcher for this instance on this port."""
        with self._lock:
            key = (instance_id, port)
            if key not in self._engines:
                self._engines[key] = EngineProgress(read, self._clock)
            return self._engines[key]
