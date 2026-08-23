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
from dataclasses import dataclass, field


class QueueFull(RuntimeError):
    """More requests are waiting than this manager will hold."""


class WillNotFit(RuntimeError):
    """This model does not fit on this machine, whatever else is unloaded.

    Told to the request rather than discovered by emptying the machine and
    failing anyway: unloading what was working, to load something that was
    never going to fit, is the worst of both.

    `detail` is filled in by whoever knows the numbers — this file does not —
    and reaches the client as fields it can act on rather than a sentence it
    would have to parse.
    """

    kind = "insufficient_memory"
    code = "model_does_not_fit"

    def __init__(self, message: str, detail: dict | None = None) -> None:
        super().__init__(message)
        self.detail = detail or {}


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
        self._make_room = make_room or (lambda shape, loaded:
                                        [item["shape"] for item in loaded])
        self.max_waiting = max_waiting

        self._lock = threading.Lock()
        self._numbers = itertools.count(1)
        # Shape -> how many requests it is answering. Being in here is what
        # "loaded" means; there is no second flag to disagree with it.
        self._loaded: dict = {}
        # Shape -> when it last had a request admitted, so the choice of what
        # to unload can prefer whatever nobody has wanted for longest.
        self._last_used: dict = {}
        self._queue: list[_Waiting] = []
        self._switching = False
        # Bumped by `reset`. A load already under way finishes after it — the
        # engine is starting and cannot be called back — and its result must
        # not be published into a world that has been thrown away since.
        self._epoch = 0

    # -- what a request does ------------------------------------------------

    def enter(self, shape, still_wanted=None) -> None:
        """Wait until this request may run. Returns once it has a place.

        Raises QueueFull if too many are already waiting, Abandoned if the
        client went away first, or whatever the load raised.
        """
        with self._lock:
            if self._admits_now(shape):
                self._take_place(shape)
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

    def leave(self, shape=None) -> None:
        """Give back a place, and let in whoever can go next.

        `shape` says which model the request was on. Without it the place is
        taken from whichever is holding one, which is right while only one is
        loaded and is why every caller predating the budget still works.
        """
        with self._lock:
            if shape is not None and self._loaded.get(shape, 0) > 0:
                self._loaded[shape] -= 1
            else:
                for held in self._loaded:
                    if self._loaded[held] > 0:
                        self._loaded[held] -= 1
                        break
        self._pump()

    def _take_place(self, shape) -> None:
        """One more request on this shape. Called holding the lock."""
        self._loaded[shape] = self._loaded.get(shape, 0) + 1
        self._last_used[shape] = time.time()

    # -- what the interface asks --------------------------------------------

    def state(self) -> dict:
        with self._lock:
            waiting = [{"shape": entry.shape,
                        "waiting_s": round(time.time() - entry.arrived, 1)}
                       for entry in self._queue]
            return {
                # One entry per loaded shape, with what it is answering and
                # how many it could. A list because what is on the machine is
                # a set — one today, more when the budget allows it.
                "loaded": [{"shape": shape,
                            "in_flight": count,
                            "places": self._places(shape),
                            "waiting": sum(1 for entry in self._queue
                                           if entry.shape == shape)}
                           for shape, count in self._loaded.items()],
                "in_flight": sum(self._loaded.values()),
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
            self._epoch += 1
            turned_away = self._queue
            self._queue = []
            self._loaded = {}
            self._last_used = {}
        for entry in turned_away:
            entry.failure = Abandoned(reason)
            entry.admitted.set()
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
            if any(self._loaded.values()) or self._queue or self._switching:
                return
            now = time.time()
            self._loaded = {shape: 0 for shape in shapes}
            self._last_used = {shape: now for shape in shapes}

    def forget_current(self) -> None:
        """Note that the machine no longer holds what it did.

        Called when something outside took a model off. The next request loads
        again rather than walking onto an empty card.
        """
        with self._lock:
            self._loaded = {}
            self._last_used = {}

    # -- the decision -------------------------------------------------------

    def _admits_now(self, shape) -> bool:
        """Straight through, or into the queue. Called holding the lock."""
        return (not self._queue
                and not self._switching
                and shape in self._loaded
                and self._loaded[shape] < self._places(shape))

    def _loaded_now(self) -> list[dict]:
        """What is on the machine, for whoever decides what has to go.

        Called holding the lock, and handed out as plain dictionaries so the
        decision can be taken outside this file — which knows nothing about
        memory and should not learn.
        """
        return [{"shape": shape, "in_flight": count,
                 "last_used": self._last_used.get(shape, 0.0)}
                for shape, count in self._loaded.items()]

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
            shape, victims, photograph, epoch = job
            failure = None
            try:
                self._switch(shape, victims)
            except BaseException as error:      # reported to everyone waiting
                failure = error
            _release(self._settle(shape, victims, photograph, failure, epoch))

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
            if head.shape in self._loaded:
                if self._loaded[head.shape] >= self._places(head.shape):
                    break                       # full: wait for a place
                self._queue.pop(0)
                if self._gone(head):
                    continue
                self._take_place(head.shape)
                ready.append(head)
                continue

            # The head wants something that is not on the machine. Who has to
            # go for it to fit? Empty means it fits beside what is there and
            # nothing is taken off.
            victims = self._make_room(head.shape, self._loaded_now())
            if victims is None:
                # It will not fit however much is unloaded. Say so to the whole
                # run rather than emptying the machine to discover it.
                self._queue.pop(0)
                if not self._gone(head):
                    head.failure = WillNotFit(
                        "there is not enough memory for this model on this "
                        "machine, whatever else is unloaded")
                    head.admitted.set()
                continue
            if any(self._loaded.get(shape, 0) for shape in victims):
                # One of them is still answering. Everything waits until it
                # finishes — including requests behind this one for models that
                # are loaded and idle. That is the ordering rule doing its job:
                # letting them past is how the request at the head never gets
                # served. And because the door is shut, nothing new joins those
                # models either, so the wait is bounded by what is already in
                # flight.
                break
            # The run at the front of the queue, and only that. Everything
            # from the first request wanting something else stays where it is,
            # however many more of this shape are behind it.
            #
            # Taking all of them would serve requests younger than the one
            # already waiting, and the whole point of oldest-first is that a
            # workflow may be held up by exactly that older request. Whether
            # somebody younger could have been served cheaply is not the
            # question; whether they arrived later is.
            #
            # Clients that have gone are dropped while the run is taken, not
            # after. After is too late: the swap has happened, the model that
            # was working is off the card, and the load was for nobody. That is
            # the fault this whole file exists to prevent, and checking a
            # moment too late reproduces it exactly.
            shape = head.shape
            photograph = []
            taken = 0
            while taken < len(self._queue) and self._queue[taken].shape == shape:
                entry = self._queue[taken]
                taken += 1
                if not self._gone(entry):
                    photograph.append(entry)
            self._queue = self._queue[taken:]
            if not photograph:
                continue                        # all gone: try the next shape
            self._switching = True
            return ready, (shape, tuple(victims), photograph, self._epoch)
        return ready, None

    def _settle(self, shape, victims: tuple, photograph: list[_Waiting],
                failure: BaseException | None, epoch: int) -> list[_Waiting]:
        """Publish what the load did, and admit who fits.

        Unless everything was thrown away while it ran. A forced stop empties
        the card and turns away everyone waiting; a load that started before it
        cannot be called back, so its result is discarded instead. The card is
        left unknown rather than claimed — whatever is on it now, the next
        request unloads before it loads.
        """
        ready: list[_Waiting] = []
        with self._lock:
            self._switching = False
            if epoch != self._epoch:
                for entry in photograph:
                    entry.failure = failure or Abandoned(
                        "everything was stopped while this model was loading")
                    entry.admitted.set()
                return ready
            # Whatever was taken off is off, whether the load that followed
            # worked or not. Claiming otherwise would send the next request to
            # a port with nothing behind it.
            for gone in victims:
                self._loaded.pop(gone, None)
                self._last_used.pop(gone, None)
            if failure is not None:
                for entry in photograph:
                    entry.failure = failure
                    entry.admitted.set()
                return ready
            self._loaded[shape] = 0
            self._last_used[shape] = time.time()
            room = self._places(shape)
            overflow = []
            for entry in photograph:
                if self._gone(entry):
                    continue
                if self._loaded[shape] >= room:
                    overflow.append(entry)
                    continue
                self._take_place(shape)
                ready.append(entry)
            # More were in the run than fit. They go back at the front, in
            # order, and are let in as places free — they are still the oldest
            # requests waiting, and putting them behind newer ones would be the
            # same unfairness by another route.
            self._queue[:0] = overflow
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
