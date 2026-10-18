"""How the scheduler chooses the next run and publishes what a load did.

The rules themselves are written down in `scheduler.py`. This is the part of
the `Scheduler` that applies them under its lock: who goes in now, which run of
requests a switch is for, and who is let in once the model is up. It is a base
class of `Scheduler` (a mixin) rather than a separate object because every step
reads and changes the same few fields, all guarded by the scheduler's one lock.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

from .scheduler_parts import AbandonedError, AdmissionWithdrawnError, Waiting, WillNotFitError

# A switch to perform once the lock is dropped: the shape to load, what to take
# off for it, the requests it is for, and the reset epoch it was decided in.
Switch = tuple[object, tuple, list[Waiting], int]

# `_plan_switch` saying: the head must wait for a request still in flight.
_WAIT = "wait"


class QueuePlanning:
    """The decisions, made holding the lock (see `Scheduler`, which owns the fields)."""

    _lock: threading.Lock
    _queue: list[Waiting]
    _loaded: dict
    _last_used: dict
    _switching: bool
    _epoch: int
    _places: Callable[[object], int]
    _make_room: Callable[[object, list[dict]], list | None]

    def _take_place(self, shape) -> None:
        """One more request on this shape. Called holding the lock."""
        self._loaded[shape] = self._loaded.get(shape, 0) + 1
        self._last_used[shape] = time.time()

    def _loaded_now(self) -> list[dict]:
        """What is on the machine, for whoever decides what has to go.

        Called holding the lock, and handed out as plain dictionaries so the
        decision can be taken outside this file — which knows nothing about
        memory and should not learn.
        """
        return [{"shape": shape, "in_flight": count,
                 "last_used": self._last_used.get(shape, 0.0)}
                for shape, count in self._loaded.items()]

    def _plan(self) -> tuple[list[Waiting], Switch | None]:
        """Decide what can happen now. Called holding the lock.

        Returns the entries to wake and, at most, one switch for the caller to
        perform after the lock is dropped. Waking happens outside the lock too:
        a thread waking up wants the lock, and handing it out while holding it
        is how a deadlock gets written.
        """
        ready: list[Waiting] = []
        while self._queue and not self._switching:
            head = self._queue[0]
            if head.shape in self._loaded:
                if not self._admit_head(head, ready):
                    break                       # full: wait for a place
                continue
            switch = self._plan_switch(head)
            if switch == _WAIT:
                break
            if isinstance(switch, tuple):
                self._switching = True
                return ready, switch
        return ready, None

    def _admit_head(self, head: Waiting, ready: list[Waiting]) -> bool:
        """The head's model is loaded: let it in if a place is free. False when full."""
        if self._loaded[head.shape] >= self._places(head.shape):
            return False
        self._queue.pop(0)
        if not self._gone(head):
            self._take_place(head.shape)
            ready.append(head)
        return True

    def _plan_switch(self, head: Waiting) -> Switch | str | None:
        """The head wants something not on the machine: the switch for it.

        `_WAIT` while something that has to go is still answering; None when
        the head was dealt with (failed, or gone) and the queue is looked at
        again.
        """
        # Who has to go for it to fit? Empty means it fits beside what is
        # there and nothing is taken off.
        victims = self._make_room(head.shape, self._loaded_now())
        if victims is None:
            # It will not fit however much is unloaded. Say so to the whole
            # run rather than emptying the machine to discover it.
            self._queue.pop(0)
            if not self._gone(head):
                head.fail(WillNotFitError(
                    "there is not enough memory for this model on this "
                    "machine, whatever else is unloaded"))
            return None
        if any(self._loaded.get(shape, 0) for shape in victims):
            # One of them is still answering. Everything waits until it
            # finishes — including requests behind this one for models that
            # are loaded and idle. That is the ordering rule doing its job:
            # letting them past is how the request at the head never gets
            # served. And because the door is shut, nothing new joins those
            # models either, so the wait is bounded by what is already in
            # flight.
            return _WAIT
        photograph = self._take_run(head.shape)
        if not photograph:
            return None                         # all gone: try the next shape
        return head.shape, tuple(victims), photograph, self._epoch

    def _take_run(self, shape) -> list[Waiting]:
        """The run at the front of the queue wanting `shape`, and only that.

        Everything from the first request wanting something else stays where
        it is, however many more of this shape are behind it. Taking all of
        them would serve requests younger than the one already waiting, and
        the whole point of oldest-first is that a workflow may be held up by
        exactly that older request.

        Clients that have gone are dropped while the run is taken, not after.
        After is too late: the swap has happened, the model that was working
        is off the card, and the load was for nobody.
        """
        photograph = []
        taken = 0
        while taken < len(self._queue) and self._queue[taken].shape == shape:
            entry = self._queue[taken]
            taken += 1
            if not self._gone(entry):
                photograph.append(entry)
        self._queue = self._queue[taken:]
        return photograph

    def _settle(self, switch: Switch, failure: BaseException | None) -> list[Waiting]:
        """Publish what the load did, and admit who fits.

        Unless everything was thrown away while it ran. A forced stop empties
        the card and turns away everyone waiting; a load that started before it
        cannot be called back, so its result is discarded instead. The card is
        left unknown rather than claimed — whatever is on it now, the next
        request unloads before it loads.
        """
        shape, victims, photograph, epoch = switch
        with self._lock:
            self._switching = False
            if epoch != self._epoch:
                for entry in photograph:
                    entry.fail(failure or AbandonedError(
                        "everything was stopped while this model was loading"))
                return []
            # Whatever was taken off is off, whether the load that followed
            # worked or not. Claiming otherwise would send the next request to
            # a port with nothing behind it.
            for gone in victims:
                self._loaded.pop(gone, None)
                self._last_used.pop(gone, None)
            if failure is not None:
                for entry in photograph:
                    entry.fail(failure)
                return []
            self._loaded[shape] = 0
            self._last_used[shape] = time.time()
            return self._admit_run(shape, photograph)

    def _admit_run(self, shape, photograph: list[Waiting]) -> list[Waiting]:
        """Let in as many of the run as the new model has places for.

        More were in the run than fit? They go back at the front, in order,
        and are let in as places free — they are still the oldest requests
        waiting, and putting them behind newer ones would be the same
        unfairness by another route.
        """
        ready: list[Waiting] = []
        overflow = []
        room = self._places(shape)
        for entry in photograph:
            if self._gone(entry):
                continue
            if self._loaded[shape] >= room:
                overflow.append(entry)
                continue
            self._take_place(shape)
            ready.append(entry)
        self._queue[:0] = overflow
        return ready

    @staticmethod
    def _gone(entry: Waiting) -> bool:
        """Whether the request is no longer to be served. Dropped, and woken.

        Asked at the last possible moment, because the expensive part is what
        comes after: a client that walked away should not cost a swap and a
        forty-second load, nor take the card off the model that was working.
        `still_wanted` may also withdraw the request with a reason of its own
        (`AdmissionWithdrawnError`, ADR 0005); the request then fails with it.
        """
        if entry.still_wanted is None:
            return False
        try:
            if entry.still_wanted():
                return False
        except AdmissionWithdrawnError as withdrawn:
            entry.fail(withdrawn)
            return True
        except Exception:
            return False                        # cannot tell: serve it
        entry.fail(AbandonedError("the client closed the connection while waiting"))
        return True
