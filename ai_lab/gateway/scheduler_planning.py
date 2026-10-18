"""How the scheduler decides: the state it keeps and the decisions taken on it.

The rules themselves are written down in `scheduler.py`. `QueueState` is what
the scheduler knows (the queue, what is loaded, whether a load runs);
`Planner` applies the rules to it — who goes in now, which run of requests a
switch is for, who is let in once the model is up. `Scheduler` owns both and
the one lock that guards the state; the planner is only ever called with that
lock held and takes none itself.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

from .scheduler_parts import AbandonedError, AdmissionWithdrawnError, Waiting, WillNotFitError

# A switch to perform once the lock is dropped: the shape to load, what to take
# off for it, the requests it is for, and the reset epoch it was decided in.
Switch = tuple[object, tuple, list[Waiting], int]

# `_plan_switch` saying: the head must wait for a request still in flight.
_WAIT = "wait"


@dataclass
class QueueState:
    """Everything the scheduler knows about the card and its queue.

    Owned by `Scheduler` and guarded by its one lock: every read and every
    change of these fields happens with that lock held. Nothing else locks it.
    """

    # Who waits, oldest first.
    queue: list[Waiting] = field(default_factory=list)
    # Shape -> how many requests it is answering. Being in here is what
    # "loaded" means; there is no second flag to disagree with it.
    loaded: dict = field(default_factory=dict)
    # Shape -> when it last had a request admitted, so the choice of what to
    # unload can prefer whatever nobody has wanted for longest.
    last_used: dict = field(default_factory=dict)
    # True while a load runs with the lock dropped, so a second thread does not
    # start a second load into it.
    switching: bool = False
    # Bumped by `reset`. A load already under way finishes after it, and its
    # result must not be published into a world thrown away since.
    epoch: int = 0


class Planner:
    """The scheduler's decisions: who goes in now, which run a switch is for, who after a load.

    Called only under the scheduler's lock. It holds no lock and no state of
    its own: every method is handed the `QueueState` to read and change, so
    the one owner of that state, and of the lock that guards it, stays
    `Scheduler`.

    `places(shape)` says how many requests a shape serves at once;
    `make_room(shape, loaded)` which loaded shapes must go for it to fit
    (empty: it fits beside them; None: it never will).
    """

    def __init__(self, places: Callable[[object], int],
                 make_room: Callable[[object, list[dict]], list | None]) -> None:
        self._places = places
        self._make_room = make_room

    def admits_now(self, state: QueueState, shape) -> bool:
        """Straight through, or into the queue: nobody waits, no load runs, a place is free."""
        return (not state.queue
                and not state.switching
                and shape in state.loaded
                and state.loaded[shape] < self._places(shape))

    def take_place(self, state: QueueState, shape) -> None:
        """One more request on this shape."""
        state.loaded[shape] = state.loaded.get(shape, 0) + 1
        state.last_used[shape] = time.time()

    def _loaded_now(self, state: QueueState) -> list[dict]:
        """What is on the machine, for whoever decides what has to go.

        Called holding the lock, and handed out as plain dictionaries so the
        decision can be taken outside this file — which knows nothing about
        memory and should not learn.
        """
        return [{"shape": shape, "in_flight": count,
                 "last_used": state.last_used.get(shape, 0.0)}
                for shape, count in state.loaded.items()]

    def plan(self, state: QueueState) -> tuple[list[Waiting], Switch | None]:
        """Decide what can happen now. Called holding the lock.

        Returns the entries to wake and, at most, one switch for the caller to
        perform after the lock is dropped. Waking happens outside the lock too:
        a thread waking up wants the lock, and handing it out while holding it
        is how a deadlock gets written.
        """
        ready: list[Waiting] = []
        while state.queue and not state.switching:
            head = state.queue[0]
            if head.shape in state.loaded:
                if not self._admit_head(state, head, ready):
                    break                       # full: wait for a place
                continue
            switch = self._plan_switch(state, head)
            if switch == _WAIT:
                break
            if isinstance(switch, tuple):
                state.switching = True
                return ready, switch
        return ready, None

    def _admit_head(self, state: QueueState, head: Waiting,
                    ready: list[Waiting]) -> bool:
        """The head's model is loaded: let it in if a place is free. False when full."""
        if state.loaded[head.shape] >= self._places(head.shape):
            return False
        state.queue.pop(0)
        if not self.gone(head):
            self.take_place(state, head.shape)
            ready.append(head)
        return True

    def _plan_switch(self, state: QueueState, head: Waiting) -> Switch | str | None:
        """The head wants something not on the machine: the switch for it.

        `_WAIT` while something that has to go is still answering; None when
        the head was dealt with (failed, or gone) and the queue is looked at
        again.
        """
        # Who has to go for it to fit? Empty means it fits beside what is
        # there and nothing is taken off.
        victims = self._make_room(head.shape, self._loaded_now(state))
        if victims is None:
            # It will not fit however much is unloaded. Say so to the whole
            # run rather than emptying the machine to discover it.
            state.queue.pop(0)
            if not self.gone(head):
                head.fail(WillNotFitError(
                    "there is not enough memory for this model on this "
                    "machine, whatever else is unloaded"))
            return None
        if any(state.loaded.get(shape, 0) for shape in victims):
            # One of them is still answering. Everything waits until it
            # finishes — including requests behind this one for models that
            # are loaded and idle. That is the ordering rule doing its job:
            # letting them past is how the request at the head never gets
            # served. And because the door is shut, nothing new joins those
            # models either, so the wait is bounded by what is already in
            # flight.
            return _WAIT
        photograph = self._take_run(state, head.shape)
        if not photograph:
            return None                         # all gone: try the next shape
        return head.shape, tuple(victims), photograph, state.epoch

    def _take_run(self, state: QueueState, shape) -> list[Waiting]:
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
        while taken < len(state.queue) and state.queue[taken].shape == shape:
            entry = state.queue[taken]
            taken += 1
            if not self.gone(entry):
                photograph.append(entry)
        state.queue = state.queue[taken:]
        return photograph

    def settle(self, state: QueueState, switch: Switch,
               failure: BaseException | None) -> list[Waiting]:
        """Publish what the load did, and admit who fits.

        Unless everything was thrown away while it ran. A forced stop empties
        the card and turns away everyone waiting; a load that started before it
        cannot be called back, so its result is discarded instead. The card is
        left unknown rather than claimed — whatever is on it now, the next
        request unloads before it loads.
        """
        shape, victims, photograph, epoch = switch
        state.switching = False
        if epoch != state.epoch:
            for entry in photograph:
                entry.fail(failure or AbandonedError(
                    "everything was stopped while this model was loading"))
            return []
        # Whatever was taken off is off, whether the load that followed
        # worked or not. Claiming otherwise would send the next request to
        # a port with nothing behind it.
        for gone in victims:
            state.loaded.pop(gone, None)
            state.last_used.pop(gone, None)
        if failure is not None:
            for entry in photograph:
                entry.fail(failure)
            return []
        state.loaded[shape] = 0
        state.last_used[shape] = time.time()
        return self._admit_run(state, shape, photograph)

    def _admit_run(self, state: QueueState, shape,
                   photograph: list[Waiting]) -> list[Waiting]:
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
            if self.gone(entry):
                continue
            if state.loaded[shape] >= room:
                overflow.append(entry)
                continue
            self.take_place(state, shape)
            ready.append(entry)
        state.queue[:0] = overflow
        return ready

    @staticmethod
    def gone(entry: Waiting) -> bool:
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
