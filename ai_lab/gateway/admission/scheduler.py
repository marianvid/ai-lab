"""Who gets the card next.

A machine holds as many models as its memory allows. Several requests may want
the same one and are served together — that is what the engines are built for.
A request wanting a model that is not loaded has to wait for room to be made,
and then for the load.

**How much room, and whose memory has to go, is not decided here.** This file
is handed a function that answers both, because a shape is an opaque key and
this knows nothing about models or megabytes. Given none, every loaded shape is
a victim of every other, which is one model at a time — the behaviour this had
before there was a budget, and what every test in this file still runs.

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

**The queue is served in order. Requests next to each other wanting the same
shape go in together.**

That is the whole rule. The oldest request decides what is loaded, and the run
of requests behind it wanting the same thing comes with it. The run stops at
the first request wanting something else — everything from there stays in the
queue, however many more of the first shape are behind it.

It is tempting to sweep up those later ones too: same model, already loaded,
free. It was written that way first. But they arrived *after* the request that
wants something else, and serving them ahead of it is the thing oldest-first
exists to prevent. A workflow can be held up by that older request, and no
amount of cheaply-served younger ones makes up for holding it longer.

The cost is real and is not hidden: requests that alternate between two models
swap on every one of them. Two models genuinely needed at once is the card's
limit, not this file's, and no ordering rule escapes it.

Requests arriving *while* a model loads wait for the next round, even for the
shape being loaded. Without that, the model just loaded starves the one that
was waiting — the door problem again with the names swapped.

There is no dwell time, no batching by size, nothing that trades fairness for
fewer loads: the fifty requests for one model may be waiting on the answer to
the one request for another.

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
from .parts import (
    AbandonedError,
    QueueFullError,
    StillWanted,
    Waiting,
    WillNotFitError,
    release,
)
from .planning import Planner, QueueState
from .replaced import Replacements

# Re-exported: callers import the scheduler's errors from here.
__all__ = ["AbandonedError", "QueueFullError", "Replacements", "Scheduler", "StillWanted",
           "WillNotFitError"]


class Scheduler:
    """The queue, the places, and the decision to swap.

    Every public method may be called from any thread: one per request, which
    is how the web server works.

    **The lock is never held while a model loads.** Loading takes between four
    seconds and a minute, and everything here would stop for it — including the
    page asking what is going on, which is exactly when somebody wants to know.
    So the decision is taken under the lock, the lock is dropped, the load
    happens, and the lock is taken again to publish the result.
    `QueueState.switching` marks the gap so a second thread does not start a
    second load into it.

    **One owner for the lock.** This class owns the lock and the `QueueState`
    it guards; the `Planner` takes its decisions on that state and is only
    ever called with the lock held here. Nothing else locks or touches it.
    """

    def __init__(self, switch, places, make_room=None,
                 max_waiting: int = 150) -> None:
        """
        `switch(shape, victims)` takes those shapes off and puts this one on.
        It may raise, and then everything waiting for that shape is failed with
        what it raised. What taking off involves is not this file's business.

        `places(shape)` says how many requests that shape serves at once.

        `make_room(shape, loaded)` says which of the loaded shapes have to go
        for this one to fit, in the order they should go. An empty list means
        it fits beside them; `None` means it never will, whatever is unloaded.
        `loaded` is one dict per loaded shape — `shape`, `in_flight`,
        `last_used` — so the decision can prefer an idle one, which costs no
        waiting, over one still answering, which costs whatever it has left.

        Without it, one shape at a time: everything loaded is a victim of
        everything else. That is what this did before there was a memory
        budget, and every existing test of this file still runs that way.
        """
        self._switch = switch
        self._places = places
        self._planner = Planner(places, make_room or (
            lambda shape, loaded: [item["shape"] for item in loaded]))
        self.max_waiting = max_waiting

        self._lock = threading.Lock()
        self._numbers = itertools.count(1)
        self._state = QueueState()

    # -- what a request does ------------------------------------------------

    def enter(self, shape, still_wanted: StillWanted | None = None) -> None:
        """Wait until this request may run. Returns once it has a place.

        Raises QueueFullError if too many are already waiting, AbandonedError
        if the client went away first (or `still_wanted` withdrew it), or
        whatever the load raised.
        """
        with self._lock:
            if self._planner.admits_now(self._state, shape):
                self._planner.take_place(self._state, shape)
                return
            if len(self._state.queue) >= self.max_waiting:
                raise QueueFullError(
                    f"{len(self._state.queue)} requests are already waiting for the "
                    f"card, which is as many as this manager will hold. Try "
                    f"again shortly.")
            entry = Waiting(number=next(self._numbers), shape=shape,
                            arrived=time.time(), still_wanted=still_wanted)
            self._state.queue.append(entry)
        # An idle card with the wrong model on it has nobody to start things
        # off but the request that just arrived.
        self._pump()
        entry.admitted.wait()
        if entry.failure is not None:
            raise entry.failure

    def leave(self, shape=None) -> None:
        """Give back a place, and let in whoever can go next.

        `shape` says which model the request was on. Without it the place is
        taken from whichever is holding one, which is right while only one is
        loaded and is why every caller predating the budget still works.
        """
        with self._lock:
            if shape is not None and self._state.loaded.get(shape, 0) > 0:
                self._state.loaded[shape] -= 1
            else:
                for held in self._state.loaded:
                    if self._state.loaded[held] > 0:
                        self._state.loaded[held] -= 1
                        break
        self._pump()

    # -- what the interface asks --------------------------------------------

    def state(self) -> dict:
        """What is loaded, how busy each model is, and who waits — for the page."""
        with self._lock:
            waiting = [{"shape": entry.shape,
                        "waiting_s": round(time.time() - entry.arrived, 1)}
                       for entry in self._state.queue]
            return {
                # One entry per loaded shape, with what it is answering and
                # how many it could. A list because what is on the machine is
                # a set — one today, more when the budget allows it.
                "loaded": [{"shape": shape,
                            "in_flight": count,
                            "places": self._places(shape),
                            "waiting": sum(1 for entry in self._state.queue
                                           if entry.shape == shape)}
                           for shape, count in self._state.loaded.items()],
                "in_flight": sum(self._state.loaded.values()),
                "switching": self._state.switching,
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
            self._state.epoch += 1
            turned_away = self._state.queue
            self._state.queue = []
            self._state.loaded = {}
            self._state.last_used = {}
        for entry in turned_away:
            entry.fail(AbandonedError(reason))
        return len(turned_away)

    def adopt(self, *shapes) -> None:
        """Take up what is already on the machine.

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
            if any(self._state.loaded.values()) or self._state.queue or self._state.switching:
                return
            now = time.time()
            self._state.loaded = dict.fromkeys(shapes, 0)
            self._state.last_used = dict.fromkeys(shapes, now)

    def forget_current(self) -> None:
        """Note that the machine no longer holds what it did.

        Called when something outside took a model off. The next request loads
        again rather than walking onto an empty card.
        """
        with self._lock:
            self._state.loaded = {}
            self._state.last_used = {}

    # -- the decision -------------------------------------------------------

    def _pump(self) -> None:
        """Move things along until nothing more can happen without waiting.

        Loops because finishing a switch may let in requests, and letting them
        in changes nothing else — but a failed switch leaves a queue that may
        now want a different shape, and that is another round.
        """
        while True:
            with self._lock:
                ready, job = self._planner.plan(self._state)
            release(ready)
            if job is None:
                return
            shape, victims, _, _ = job
            failure = None
            try:
                self._switch(shape, victims)
            except BaseException as error:      # reported to everyone waiting
                failure = error
            with self._lock:
                ready = self._planner.settle(self._state, job, failure)
            release(ready)
