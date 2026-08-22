"""Who gets the card next.

One card holds one model. Several requests may want the same one, and they can
be served together — that is what the engines are built for. Requests wanting a
different model have to wait for the card to empty, and then for the load.

This file is the policy and nothing else. It knows nothing about models,
engines, ports or HTTP: a "shape" is an opaque key, and swapping to one is a
callable it is handed. That is what lets the rules below be tested without a
model anywhere near them.

## The rules

A request goes straight through only if all three hold:

1. nothing is waiting,
2. it wants the shape already on the card,
3. there is a free place — the limit is the engine's own, declared per model.

Otherwise it joins the back of the queue. No exceptions and no jumping.

Condition 1 is the one that matters. The moment anybody is waiting, the door
closes for everybody. Without it a busy model never goes idle, so the switch
never happens, and a request for another model waits for ever. Under continuous
load that is not unlikely — it is certain.

When a request finishes it frees a place, and then:

- the head of the queue wants the current shape and there is room — it goes in,
  and again for as many as fit;
- the head wants something else — nobody goes in. The card has to empty first;
- the card is empty — switch.

## Switching

The **oldest waiting request wins**, and its shape is loaded. Then everything
in the queue wanting that same shape *at that moment* goes in together — a
photograph. Requests arriving afterwards wait for the next round, even for the
shape just loaded. Without that, the model that has just been loaded starves
the one that was waiting, which is the first problem again with the names
swapped.

Oldest-wins is strict. There is no dwell time, no batching by size, nothing
that trades fairness for fewer loads — because the fifty requests for one model
may be waiting on the answer to the one request for another, and serving them
first would be optimising the workflow into a standstill.

The consequence to design workflows around: **a request must not wait, inside
itself, on another request to this same gateway.** Fifty in-flight requests all
waiting for a fifty-first would fill every place with things that cannot finish.
Fairness does not save you from that; nothing does.

## When a load fails

Every request in that photograph is failed with the engine's own words, and the
scheduler moves on to the next shape in the queue. It does not retry: a model
that would not fit a moment ago will not fit now.
"""

from __future__ import annotations

import itertools
import threading
import time
from dataclasses import dataclass, field


class QueueFull(RuntimeError):
    """More requests are waiting than this manager will hold."""


class Abandoned(RuntimeError):
    """The client went away before its turn came."""


@dataclass(slots=True)
class _Waiting:
    """One request, from arriving to being let in."""

    number: int                       # arrival order, and the tie-break
    shape: object                     # opaque; equal shapes may run together
    arrived: float
    admitted: threading.Event = field(default_factory=threading.Event)
    # Set when this one will never be admitted, and why.
    failure: BaseException | None = None
    # Asked before it is let in. A client that has gone is dropped rather than
    # served, which matters most when serving it means a forty-second load.
    still_wanted: object = None


class Scheduler:
    """The queue, the places, and the decision to swap.

    Every public method may be called from any thread: one per request, which
    is how the web server works.

    **The lock is never held while a model loads.** Loading takes between four
    seconds and a minute, and everything here would stop for it — including the
    page asking what is going on, which is exactly when somebody wants to know.
    So the decision is taken under the lock, the lock is dropped, the load
    happens, and the lock is taken again to publish the result. `_switching`
    marks the gap so a second thread does not start a second load into it.
    """

    def __init__(self, switch, places, max_waiting: int = 150) -> None:
        """
        `switch(shape)` empties the card and puts that shape on it. It may
        raise, and then everything waiting for that shape is failed with what
        it raised. What emptying involves is not this file's business.

        `places(shape)` says how many requests that shape serves at once.
        """
        self._switch = switch
        self._places = places
        self.max_waiting = max_waiting

        self._lock = threading.Lock()
        self._numbers = itertools.count(1)
        self._queue: list[_Waiting] = []
        self._in_flight = 0
        self._current: object = None
        self._loaded = False
        self._switching = False

    # -- what a request does ------------------------------------------------

    def enter(self, shape, still_wanted=None) -> None:
        """Wait until this request may run. Returns once it has a place.

        Raises QueueFull if too many are already waiting, Abandoned if the
        client went away first, or whatever the load raised.
        """
        with self._lock:
            if self._admits_now(shape):
                self._in_flight += 1
                return
            if len(self._queue) >= self.max_waiting:
                raise QueueFull(
                    f"{len(self._queue)} requests are already waiting for the "
                    f"card, which is as many as this manager will hold. Try "
                    f"again shortly.")
            entry = _Waiting(number=next(self._numbers), shape=shape,
                             arrived=time.time(), still_wanted=still_wanted)
            self._queue.append(entry)
        # An idle card with the wrong model on it has nobody to start things
        # off but the request that just arrived.
        self._pump()
        entry.admitted.wait()
        if entry.failure is not None:
            raise entry.failure

    def leave(self) -> None:
        """Give back a place, and let in whoever can go next."""
        with self._lock:
            if self._in_flight > 0:
                self._in_flight -= 1
        self._pump()

    # -- what the interface asks --------------------------------------------

    def state(self) -> dict:
        with self._lock:
            waiting = [{"shape": entry.shape,
                        "waiting_s": round(time.time() - entry.arrived, 1)}
                       for entry in self._queue]
            return {
                "current": self._current if self._loaded else None,
                "in_flight": self._in_flight,
                "places": self._places(self._current) if self._loaded else 0,
                "switching": self._switching,
                "waiting": waiting,
            }

    def reset(self, reason: str) -> int:
        """Throw everything away: nothing running, nobody waiting.

        What a forced stop means. The card is emptied by whoever forced it; this
        is the bookkeeping catching up, and every waiting request is refused
        rather than left holding a place in a queue that no longer describes
        anything. Returns how many were turned away.
        """
        with self._lock:
            turned_away = self._queue
            self._queue = []
            self._in_flight = 0
            self._current = None
            self._loaded = False
        for entry in turned_away:
            entry.failure = Abandoned(reason)
            entry.admitted.set()
        return len(turned_away)

    def adopt(self, shape) -> None:
        """Take up what is already on the card.

        systemd keeps the engines running across a manager restart, which is
        the reason for using it, so a manager coming back finds its model still
        answering. Believing the card is empty would reload it — a minute of
        work to arrive where it already was, and taking it away from whoever
        was using it.

        Ignored while anything is running or waiting here: then this scheduler
        already knows what is on the card, and a stale reading from outside
        must not overwrite it.
        """
        with self._lock:
            if self._in_flight or self._queue or self._switching:
                return
            self._current = shape
            self._loaded = True

    def forget_current(self) -> None:
        """Note that the card no longer holds what it did.

        Called when something outside took the model off. The next request has
        to load again rather than walk onto an empty card.
        """
        with self._lock:
            self._loaded = False
            self._current = None

    # -- the decision -------------------------------------------------------

    def _admits_now(self, shape) -> bool:
        """Straight through, or into the queue. Called holding the lock."""
        return (not self._queue
                and not self._switching
                and self._loaded
                and shape == self._current
                and self._in_flight < self._places(shape))

    def _pump(self) -> None:
        """Move things along until nothing more can happen without waiting.

        Loops because finishing a switch may let in requests, and letting them
        in changes nothing else — but a failed switch leaves a queue that may
        now want a different shape, and that is another round.
        """
        while True:
            with self._lock:
                ready, job = self._plan()
            _release(ready)
            if job is None:
                return
            shape, photograph = job
            failure = None
            try:
                self._switch(shape)
            except BaseException as error:      # reported to everyone waiting
                failure = error
            _release(self._settle(shape, photograph, failure))

    def _plan(self) -> tuple[list[_Waiting], tuple | None]:
        """Decide what can happen now. Called holding the lock.

        Returns the entries to wake and, at most, one switch for the caller to
        perform after the lock is dropped. Waking happens outside the lock too:
        a thread waking up wants the lock, and handing it out while holding it
        is how a deadlock gets written.
        """
        ready: list[_Waiting] = []
        while self._queue and not self._switching:
            head = self._queue[0]
            if self._loaded and head.shape == self._current:
                if self._in_flight >= self._places(self._current):
                    break                       # full: wait for a place
                self._queue.pop(0)
                if self._gone(head):
                    continue
                self._in_flight += 1
                ready.append(head)
                continue
            if self._in_flight:
                break                           # must empty before swapping
            # The photograph is taken now, before the load. Whatever arrives
            # while a model is loading belongs to the next round, even if it
            # wants the shape being loaded — otherwise the model just loaded
            # starves the one that was waiting, which is the first problem
            # again with the names swapped.
            #
            # Clients that have gone are dropped while it is taken, not after.
            # After is too late: the swap has happened, the model that was
            # working is off the card, and the load was for nobody. That is the
            # fault this whole file exists to prevent, and checking a moment
            # too late reproduces it exactly.
            shape = head.shape
            photograph, remaining = [], []
            for entry in self._queue:
                if entry.shape != shape:
                    remaining.append(entry)
                elif not self._gone(entry):
                    photograph.append(entry)
            self._queue = remaining
            if not photograph:
                continue                        # all gone: try the next shape
            self._switching = True
            self._loaded = False
            self._current = None
            return ready, (shape, photograph)
        return ready, None

    def _settle(self, shape, photograph: list[_Waiting],
                failure: BaseException | None) -> list[_Waiting]:
        """Publish what the load did, and admit who fits."""
        ready: list[_Waiting] = []
        with self._lock:
            self._switching = False
            if failure is not None:
                for entry in photograph:
                    entry.failure = failure
                    entry.admitted.set()
                return ready
            self._current = shape
            self._loaded = True
            room = self._places(shape)
            for entry in photograph:
                if self._gone(entry):
                    continue
                if self._in_flight >= room:
                    self._queue.append(entry)   # more than fit: next round
                    continue
                self._in_flight += 1
                ready.append(entry)
        return ready

    @staticmethod
    def _gone(entry: _Waiting) -> bool:
        """Whether the client has given up. Dropped rather than served.

        Asked at the last possible moment, because the expensive part is what
        comes after: a client that walked away should not cost a swap and a
        forty-second load, nor take the card off the model that was working.
        """
        if entry.still_wanted is None:
            return False
        try:
            if entry.still_wanted():
                return False
        except Exception:
            return False                        # cannot tell: serve it
        entry.failure = Abandoned("the client closed the connection while waiting")
        entry.admitted.set()
        return True


def _release(ready: list[_Waiting]) -> None:
    for entry in ready:
        entry.admitted.set()
