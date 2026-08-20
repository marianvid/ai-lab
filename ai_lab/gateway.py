"""Making one card look like a rack of models.

An agent workflow talks to several models: one reads the code, another writes
it, a third reviews it. Each of those is a separate entry here, on its own port,
and only some of them are loaded at any moment. An agent that asks for a model
which is not running gets a refused connection and the workflow stops.

This module removes that. A request names a model; if that model is already
loaded the request goes straight to it, and if it is not, the model is loaded
first — evicting another one only if there is no room. The agent waits longer
for that first request and sees nothing else.

**What it cannot do.** One card holds what fits and no more. Two agents on this
machine asking for two models that cannot be resident together will take turns,
not run in parallel. That is a property of the hardware, not of this code, and
the only way around it is a second machine.

There is no HTTP in this file. It decides *which* entry serves a name and makes
sure it is running; forwarding the request is the web layer's job.
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field

from .operations import Operations

# The message the runtime produces when a model will not fit. Matching it is
# how this module discovers that it must evict something: the fit check already
# exists there, and duplicating the arithmetic here would mean two versions of
# it that could disagree.
DOES_NOT_FIT = re.compile(r"needs about .* free on the card", re.S)


class NotConfigured(KeyError):
    """No entry serves that model name."""


class CouldNotLoad(RuntimeError):
    """The entry exists but will not start."""


@dataclass(slots=True)
class Lease:
    """Permission to send one request to a loaded model.

    Held for as long as the request is in flight, including a streamed answer
    that takes a minute to arrive. A swap waits for every lease to be released
    before it unloads anything, which is what stops a model disappearing
    underneath a response that is still being written.
    """

    gateway: "Gateway"
    instance_id: str
    port: int

    def __enter__(self) -> "Lease":
        return self

    def __exit__(self, *_exception) -> None:
        self.gateway.release()


@dataclass(slots=True)
class _Counters:
    requests: int = 0
    swaps: int = 0
    evictions: int = 0
    waited_s: float = 0.0
    last_error: str = ""
    history: list[dict] = field(default_factory=list)


class Gateway:
    """Routes by model name, loading and evicting as needed.

    Everything is guarded by one condition variable. Requests for a model that
    is already loaded run concurrently; a swap is exclusive and waits for the
    in-flight requests to drain first.
    """

    # Keeping a record of every swap is what makes thrashing visible. An agent
    # alternating between two models pays a load on every turn, and the only
    # way to notice is to count.
    HISTORY = 50

    def __init__(self, operations: Operations) -> None:
        self.operations = operations
        self._condition = threading.Condition()
        self._inflight = 0
        self._swapping = False
        self._used_at: dict[str, float] = {}
        self.counters = _Counters()

    # -- what a client can ask for -----------------------------------------

    def catalogue(self) -> list[dict]:
        """Every configured entry, loaded or not, with the names it answers to.

        A client should not have to know which of an entry's three names to
        use, so all of them are listed and all of them are accepted.
        """
        rows = []
        for instance in self.operations.instances():
            rows.append({
                "id": instance["id"],
                "name": instance["name"],
                "model_id": instance["model_id"],
                "engine": instance["engine"],
                "port": instance["port"],
                "loaded": bool(instance["running"]),
                "ready": bool(instance["ready"]),
                "aliases": self._aliases(instance),
            })
        return rows

    @staticmethod
    def _aliases(instance: dict) -> list[str]:
        """The names one entry answers to.

        The entry id, the label a person gave it, and the last segment of the
        model path — which is what the engine reports as its own model name, so
        it is what a client copying from `/v1/models` on the engine will send.
        """
        model_id = instance.get("model_id") or ""
        names = [instance["id"], instance.get("name") or "", model_id,
                 model_id.rsplit("/", 1)[-1]]
        seen, unique = set(), []
        for name in names:
            key = name.strip().lower()
            if key and key not in seen:
                seen.add(key)
                unique.append(name.strip())
        return unique

    def resolve(self, wanted: str) -> dict:
        """The entry that serves this name, or NotConfigured."""
        wanted_key = (wanted or "").strip().lower()
        for instance in self.operations.instances():
            if any(alias.lower() == wanted_key for alias in self._aliases(instance)):
                return instance
        known = sorted({alias for instance in self.operations.instances()
                        for alias in self._aliases(instance)})
        raise NotConfigured(
            f"No configured model answers to {wanted!r}. Known: {', '.join(known)}")

    # -- getting one loaded -------------------------------------------------

    def acquire(self, wanted: str) -> Lease:
        """Ensure the named model is loaded, and hold a lease on it.

        Blocks while another request is swapping, and blocks for the length of
        a load if this request is the one that triggers it.
        """
        started = time.perf_counter()
        instance = self.resolve(wanted)
        instance_id, port = instance["id"], instance["port"]

        while True:
            with self._condition:
                while self._swapping:
                    self._condition.wait()
                if self._is_ready(instance_id):
                    self._inflight += 1
                    self._used_at[instance_id] = time.time()
                    self.counters.requests += 1
                    self.counters.waited_s += time.perf_counter() - started
                    return Lease(self, instance_id, port)
                # This request has to bring the model up. Claim that job, then
                # wait for the requests already running to finish, because
                # loading may have to evict the model they are talking to.
                self._swapping = True
                while self._inflight > 0:
                    self._condition.wait()

            try:
                self._bring_up(instance_id)
            except Exception as error:
                with self._condition:
                    self._swapping = False
                    self.counters.last_error = str(error)
                    self._condition.notify_all()
                raise
            with self._condition:
                self._swapping = False
                self._condition.notify_all()
            # Round again rather than assuming: between finishing the load and
            # taking the lease, nothing else can have run, but checking is
            # cheaper than reasoning about it.

    def release(self) -> None:
        with self._condition:
            self._inflight -= 1
            self._condition.notify_all()

    # -- internals ----------------------------------------------------------

    def _is_ready(self, instance_id: str) -> bool:
        for instance in self.operations.instances():
            if instance["id"] == instance_id:
                return bool(instance["ready"])
        return False

    def _bring_up(self, instance_id: str) -> None:
        """Load one entry, evicting the least recently used until it fits.

        The runtime already refuses a load that will not fit and says how much
        it needed. That refusal is the fit test: try, and if the answer is that
        it does not fit, free the oldest model and try again.
        """
        evicted: list[str] = []
        while True:
            operation = self.operations.load(instance_id)
            if operation.ok:
                self.counters.swaps += 1
                self._record(instance_id, operation.total_ms, evicted)
                return
            if not DOES_NOT_FIT.search(operation.error or ""):
                raise CouldNotLoad(operation.error or f"{instance_id} would not start")
            victim = self._eviction_candidate(instance_id)
            if victim is None:
                raise CouldNotLoad(operation.error)
            self.operations.unload(victim)
            self.counters.evictions += 1
            evicted.append(victim)

    def _eviction_candidate(self, keep: str) -> str | None:
        """The loaded entry used longest ago, or None if nothing is loaded.

        Least recently used rather than anything cleverer: in an agent workflow
        the model that has not been asked for in a while is the one least likely
        to be asked for next.
        """
        running = [instance["id"] for instance in self.operations.instances()
                   if instance["running"] and instance["id"] != keep]
        if not running:
            return None
        return min(running, key=lambda name: self._used_at.get(name, 0.0))

    def _record(self, instance_id: str, took_ms: int, evicted: list[str]) -> None:
        """One line per completed swap, naming what had to go to make room.

        Written after the load succeeds rather than as each eviction happens,
        so a reader sees a single entry per swap instead of one per victim.
        """
        self.counters.history.append({
            "at": time.time(), "loaded": instance_id,
            "evicted": list(evicted), "took_ms": took_ms,
        })
        del self.counters.history[:-self.HISTORY]

    # -- what the interface shows ------------------------------------------

    def stats(self) -> dict:
        """Enough to see thrashing without reading a log.

        A swap count close to the request count means the workflow is changing
        model on almost every turn, which on this hardware is the difference
        between seconds and minutes per step.
        """
        with self._condition:
            counters = self.counters
            return {
                "requests": counters.requests,
                "swaps": counters.swaps,
                "evictions": counters.evictions,
                "in_flight": self._inflight,
                "swapping": self._swapping,
                "average_wait_s": round(
                    counters.waited_s / counters.requests, 3) if counters.requests else 0.0,
                "last_error": counters.last_error,
                "recent": list(reversed(counters.history[-10:])),
            }
