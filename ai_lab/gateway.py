"""One address for an agent workflow that uses several models.

An agent asks the researcher model to read, the developer model to write, the
reviewer model to check. Each is a separate entry here on its own port, and only
one of them can be on the card. An agent that names a model which is not running
gets a refused connection, and the workflow stops there.

This removes that. A request names a model; if it is already loaded the request
goes through, and if it is not, the card is emptied and that model is loaded
first. The agent waits longer for that one request and sees nothing else.

## One model at a time, and requests one at a time

That is the design, not a limitation being worked around.

The models here are chosen in advance and known to fit. The card holds one of
them at a time, and an agent workflow is a sequence — read, then write, then
check — so there is nothing to gain from overlapping requests and a great deal
to lose from two of them arriving during a swap. Every request takes the card in
turn, and holds it until its answer has been fully written.

The consequence is worth stating plainly: **two agents on this machine do not
run in parallel.** The second waits for the first. Running them at the same time
needs a second machine.

## Emptying the card properly

Switching does not unload the outgoing model and start the next one straight
away. The driver returns VRAM a moment after a process exits, and starting a
model on top of memory that has not come back yet fails in a way that reads like
the new model being too large.

So a switch unloads everything running, waits for the card to actually go quiet,
and only then loads. If it does not go quiet, the switch fails and says so,
rather than loading into a mess.

There is no HTTP in this file. It decides which entry serves a name and makes
sure it is the one on the card; forwarding the request is the web layer's job.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from .operations import Operations


class NotConfigured(KeyError):
    """No entry serves that model name."""


class CouldNotLoad(RuntimeError):
    """The entry exists but the card could not be made ready for it."""


# How quiet the card has to be before a new model is loaded, and how long to
# wait for that. The threshold is not zero because a driver keeps a small
# allocation of its own; on this machine an idle card reads about 2 MiB.
QUIET_MB = 512.0
QUIET_TIMEOUT_S = 60.0
QUIET_POLL_S = 0.5


@dataclass(slots=True)
class Lease:
    """The card, held for the length of one request.

    Taken before the request is forwarded and released after the last byte of
    the answer, including a streamed answer that takes a minute. Nothing else
    touches the card in between.
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
    switches: int = 0
    waited_s: float = 0.0
    switch_s: float = 0.0
    last_error: str = ""
    history: list[dict] = field(default_factory=list)


class Gateway:
    """Routes by model name and puts that model on the card if it is not there."""

    HISTORY = 50

    def __init__(self, operations: Operations,
                 quiet_mb: float = QUIET_MB,
                 quiet_timeout_s: float = QUIET_TIMEOUT_S,
                 poll_s: float = QUIET_POLL_S) -> None:
        self.operations = operations
        self.quiet_mb = quiet_mb
        self.quiet_timeout_s = quiet_timeout_s
        self.poll_s = poll_s
        # One card, one request at a time. A plain lock says exactly that; a
        # reader-writer arrangement would only be pretending there is a case
        # where two requests should overlap.
        self._card = threading.Lock()
        # Guards the flag below. A request can be reported as finished from two
        # places at once — the forwarding code hands the card back when the
        # answer ends, and again if the connection broke on the way out.
        self._bookkeeping = threading.Lock()
        self._held = False
        self._current: str | None = None
        self.counters = _Counters()

    # -- what a client can ask for -----------------------------------------

    def catalogue(self) -> list[dict]:
        """Every configured entry, loaded or not, with the names it answers to.

        Entries that are not loaded are listed too. A client is meant to be able
        to ask for one of them — that is the whole point.
        """
        return [{
            "id": instance["id"],
            "name": instance["name"],
            "model_id": instance["model_id"],
            "engine": instance["engine"],
            "port": instance["port"],
            "loaded": bool(instance["running"]),
            "ready": bool(instance["ready"]),
            "aliases": self._aliases(instance),
        } for instance in self.operations.instances()]

    @staticmethod
    def _aliases(instance: dict) -> list[str]:
        """The names one entry answers to.

        Its id, the label a person gave it, the model path, and the last segment
        of that path — which is the name the engine reports for itself, so it is
        what a client that read `/v1/models` from the engine will send.
        """
        model_id = instance.get("model_id") or ""
        candidates = [instance["id"], instance.get("name") or "", model_id,
                      model_id.rsplit("/", 1)[-1]]
        seen, unique = set(), []
        for name in candidates:
            key = name.strip().lower()
            if key and key not in seen:
                seen.add(key)
                unique.append(name.strip())
        return unique

    def resolve(self, wanted: str) -> dict:
        """The entry serving this name, or NotConfigured naming what is known."""
        key = (wanted or "").strip().lower()
        instances = self.operations.instances()
        for instance in instances:
            if any(alias.lower() == key for alias in self._aliases(instance)):
                return instance
        known = sorted({alias for instance in instances
                        for alias in self._aliases(instance)})
        raise NotConfigured(
            f"No configured model answers to {wanted!r}. Known: {', '.join(known)}")

    # -- taking the card ----------------------------------------------------

    def acquire(self, wanted: str) -> Lease:
        """Take the card with the named model on it.

        Waits for the request in front, then switches models if this one is not
        already loaded. Returns once the model is answering.
        """
        started = time.perf_counter()
        instance = self.resolve(wanted)
        instance_id, port = instance["id"], instance["port"]

        self._card.acquire()
        try:
            if not self._is_ready(instance_id):
                self._switch_to(instance_id)
            self._current = instance_id
            self._held = True
            self.counters.requests += 1
            self.counters.waited_s += time.perf_counter() - started
            return Lease(self, instance_id, port)
        except Exception as error:
            self.counters.last_error = str(error)
            self._card.release()
            raise

    def release(self) -> None:
        """Give the card back to whoever is waiting.

        Safe to call twice for the same request: the second call does nothing.
        Without that, a request that failed while being forwarded would hand
        back the card twice, and the second time it would be taking it away
        from the request that had already started.
        """
        with self._bookkeeping:
            if not self._held:
                return
            self._held = False
            self._card.release()

    # -- switching ----------------------------------------------------------

    def _switch_to(self, instance_id: str) -> None:
        """Empty the card, wait for it to go quiet, then load.

        Everything running is unloaded, not only the entry that was last asked
        for: a manager restart or a machine boot can leave more than one engine
        up, and loading into whatever is left is how a curated model that is
        known to fit fails to fit.
        """
        started = time.perf_counter()
        unloaded = self._clear()
        self._wait_until_quiet()

        operation = self.operations.load(instance_id)
        if not operation.ok:
            raise CouldNotLoad(operation.error or f"{instance_id} would not start")

        took = time.perf_counter() - started
        self.counters.switches += 1
        self.counters.switch_s += took
        self.counters.history.append({
            "at": time.time(), "loaded": instance_id, "unloaded": unloaded,
            "took_s": round(took, 1), "load_ms": operation.total_ms,
        })
        del self.counters.history[:-self.HISTORY]

    def _clear(self) -> list[str]:
        """Unload every running engine. Returns what was stopped."""
        stopped = []
        for instance in self.operations.instances():
            if instance["running"]:
                self.operations.unload(instance["id"])
                stopped.append(instance["id"])
        self._current = None
        return stopped

    def _wait_until_quiet(self) -> None:
        """Wait for the driver to hand the memory back.

        A process exits before its VRAM is released. Loading in that gap fails
        with a message about the model being too large, which sends whoever
        reads it looking in the wrong place entirely.
        """
        snapshot = self.operations.host.accelerator()
        if snapshot.memory_kind != "dedicated":
            return                      # unified memory: nothing to wait for
        deadline = time.perf_counter() + self.quiet_timeout_s
        while True:
            used = self.operations.host.accelerator().memory_used_mb
            if used <= self.quiet_mb:
                return
            if time.perf_counter() > deadline:
                raise CouldNotLoad(
                    f"The card still holds {used:.0f} MB {self.quiet_timeout_s:.0f} "
                    f"seconds after everything was unloaded. Something outside "
                    f"AI-Lab is using it, or an engine did not exit.")
            time.sleep(self.poll_s)

    # -- internals ----------------------------------------------------------

    def _is_ready(self, instance_id: str) -> bool:
        for instance in self.operations.instances():
            if instance["id"] == instance_id:
                return bool(instance["ready"])
        return False

    # -- what the interface shows ------------------------------------------

    def stats(self) -> dict:
        """Enough to see thrashing without reading a log.

        A switch count close to the request count means the workflow changes
        model on almost every step. Each switch is an unload, a wait and a load,
        so that is the difference between a workflow that runs and one that
        spends its time loading.
        """
        counters = self.counters
        return {
            "current": self._current,
            "busy": self._held,
            "requests": counters.requests,
            "switches": counters.switches,
            "average_wait_s": round(counters.waited_s / counters.requests, 2)
                              if counters.requests else 0.0,
            "average_switch_s": round(counters.switch_s / counters.switches, 1)
                                if counters.switches else 0.0,
            "total_switch_s": round(counters.switch_s, 1),
            "last_error": counters.last_error,
            "recent": list(reversed(counters.history[-10:])),
        }
