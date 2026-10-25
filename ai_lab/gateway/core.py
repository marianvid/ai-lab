"""Model routing, admission and load coordination for agent requests.

Scheduling, memory accounting, request state and telemetry live in their
respective modules. The gateway holds its collaborators: the `Catalogue`
(names, listings, request shapes), the `CardLoader` (making room and loading,
with the memory reading it owns) and the `Scheduler` (the queue and its lock).
See docs/gateway-behavior.md for the operating contract.
"""

from __future__ import annotations

import time
from dataclasses import replace

from .catalogue import Catalogue
from .control import GatewayControl
from .errors import (
    CardBusyError,
    CouldNotLoadError,
    ModelNotLoadedError,
    NotConfiguredError,
    ShapeNotServedError,
)
from .limits import GatewayLimits
from .loading import CardLoader, CardMemory
from .progress import EngineWatches
from .resources import GatewayResources, ask_engine
from .state import Lease, Shape
from .stats import GatewayCounters, GatewayStats, StatsSources
from .card_operations import CardOperations
from .admission.scheduler import (AbandonedError, Replacements, Scheduler, StillWanted,
                                  WillNotFitError)
from ..types import Task

# The gateway's errors are part of its interface: callers import them from here.
__all__ = ["CardBusyError", "CouldNotLoadError", "Gateway", "GatewayLimits",
           "ModelNotLoadedError", "NotConfiguredError", "ShapeNotServedError"]


class Gateway:
    """Routes by model name and puts that model on the card if it is not there.

    The queueing itself is `scheduler.Scheduler`, which knows nothing about
    models: it is handed a way to put a shape on the card and a way to ask how
    many requests that shape serves at once. Everything about names, engines
    and settings is here; everything about who goes next is there.

    Its collaborators are objects it holds, each with its own state: the
    `Catalogue` (none), the `CardLoader` (the memory reading and the sweep
    flag) and the `Scheduler` (the queue, guarded by the scheduler's own
    lock). `memory` is the machine's memory (`CardMemory`); without one the
    gateway builds a `GatewayResources` from the operations and the limits.
    """

    def __init__(self, operations: CardOperations, limits: GatewayLimits | None = None,
                 memory: CardMemory | None = None) -> None:
        limits = limits or GatewayLimits()
        self._operations = operations
        self.quiet_mb = limits.quiet_mb
        self.quiet_timeout_s = limits.quiet_timeout_s
        self.poll_s = limits.poll_s
        self.first_byte_s = limits.first_byte_s
        self.between_bytes_s = limits.between_bytes_s
        # Per-task overrides of the two waits above, keyed by `Task.value`.
        # Missing keys, or a task with no override, fall back to the defaults.
        self.task_timeouts = dict(limits.task_timeouts)
        self.max_upload_bytes = limits.max_upload_bytes
        self.max_upload_pixels = limits.max_upload_pixels
        self.max_upload_dimension = limits.max_upload_dimension
        # Stall and safety limits and the default answer cap (`limits.Patience`).
        self.patience = limits.patience
        self._watches = EngineWatches(self._restart_stalled)
        self._memory: CardMemory = memory or GatewayResources(
            operations, limits.quiet_mb, limits.quiet_timeout_s, limits.poll_s)
        self._counters = GatewayCounters()
        self._catalogue = Catalogue(operations)
        self._scheduler = Scheduler(self._switch, self._places, self._make_room,
                                   max_waiting=limits.max_waiting)
        self._loader = CardLoader(operations, self._memory, self._counters,
                                  self._scheduler.state, self._catalogue.engine_name_of)
        self.control = GatewayControl(self._scheduler)
        self.replacements = Replacements()    # taken off for another (A5, A7), and the log (A9)

    # -- what a client can ask for -----------------------------------------

    def catalogue(self) -> list[dict]:
        """Every configured entry, loaded or not, with the names it answers to."""
        return self._catalogue.catalogue()

    def describe(self, wanted: str) -> dict:
        """Everything a client needs to know about one model before using it."""
        return self._catalogue.describe(wanted)

    def resolve(self, wanted: str, instances: list[dict] | None = None) -> dict:
        """The entry with this id, or NotConfiguredError naming what is known."""
        return self._catalogue.resolve(wanted, instances)

    def engine_name_of(self, shape) -> str:
        """The display name of the engine a loaded shape runs on, for the page."""
        return self._catalogue.engine_name_of(shape)

    # -- taking a place on the card -----------------------------------------

    def acquire(self, wanted: str, shape: str | None = None,
                settings: dict | None = None,
                still_wanted: StillWanted | None = None) -> Lease:
        """Take a place on the card, with the named model on it.

        Returns once the model is answering and this request may go through.
        It may return at once — the model is loaded and there is room — or after
        waiting for a place, or after waiting for a switch and the load.

        `shape` is the kind of request about to be forwarded, and `settings`
        what the model should be started with instead of its entry's own; both
        are checked before anything is queued, so a wrong one comes back at
        once rather than after a forty-second load. Two requests wanting
        different settings cannot share a card, whatever the model.

        `still_wanted()` says whether the request is still to be served, asked
        at the head of the queue before anything is unloaded: a client that
        gave up must not cost a swap.
        """
        started = time.perf_counter()
        instance, asked_for = self._what_is_asked(wanted, shape, settings)
        self._enter(Shape.of(instance["id"], asked_for), asked_for, still_wanted)
        try:
            self._counters.requests += 1
            self._counters.waited_s += time.perf_counter() - started
            self._counters.arrivals.append((time.time(), instance["id"]))
            return self._lease(instance, asked_for)
        except BaseException:
            # The place has been taken. Anything that goes wrong between there
            # and handing it to the caller has to give it back, or the card
            # loses a place with nobody using it.
            self._scheduler.leave()
            raise

    def where_loaded(self, wanted: str, shape: str) -> tuple[int, str]:
        """Port and engine's name of a loaded model, taking no place (`GatewayControl.where_loaded`)."""
        self.adopt_what_is_there()
        return self.control.where_loaded(self._catalogue, self._operations.configured(),
                                          wanted, shape)

    def _lease(self, instance: dict, asked_for: dict) -> Lease:
        """The place handed to the caller, with what forwarding must know."""
        return Lease(self, instance["id"], instance["port"], Catalogue.engine_name(instance),
                     drop=self._catalogue.refused_fields(instance, asked_for),
                     text=Catalogue.writes_text(instance),
                     progress=self._watches.of(instance, self._catalogue.progress_reader(instance)),
                     started=time.perf_counter())

    def _what_is_asked(self, wanted: str, shape: str | None,
                       settings: dict | None) -> tuple[dict, dict]:
        """The entry asked for and the full settings it must run with.

        The configuration, not the supervisor. Which entry answers to a name,
        which engine runs it, what settings it has — all of that is the file,
        at 0.05 ms. Asking what every instance is *doing* costs 73 ms on the
        container, and this path needs that only when something outside may
        have changed the card.

        Always the full settings, never only what was asked for. A request
        naming no settings and one naming exactly the configured ones are
        asking for the same thing, and comparing partial dictionaries would
        make them different — a reload for nothing, on every other request.
        """
        entries = self._operations.configured()
        instance = self.resolve(wanted, entries)
        if shape is not None:
            self._catalogue.refuse_wrong_shape(instance, shape, entries)
        return instance, self._operations.effective_params(instance["id"], settings or {})

    def _enter(self, requested: Shape, asked_for: dict,
               still_wanted: StillWanted | None) -> None:
        """Wait in the scheduler's queue until this shape has a place."""
        self.adopt_what_is_there()
        # Some engines load weights only on their first inference. The card
        # reading taken when the server started is then too optimistic for the
        # next model. Refresh only for a switch; warm requests stay cheap.
        if not any(item["shape"] == requested
                   for item in self._scheduler.state()["loaded"]):
            self._loader.read_the_card()
        try:
            self._scheduler.enter(requested, still_wanted=still_wanted)
        except WillNotFitError as error:
            # The scheduler knows the ordering and nothing about memory, so the
            # numbers are attached here. A client that can read them can correct
            # itself — a smaller context, a smaller share of the card — without
            # anybody parsing an English sentence.
            error.detail = {"ai_lab": self._loader.why_it_does_not_fit(
                requested.instance_id, asked_for)}
            self._counters.last_error = str(error)
            raise
        except Exception as error:
            # A client that gave up is not a fault of this machine, so it does not become the error
            # the page shows.
            if not isinstance(error, AbandonedError):
                self._counters.last_error = str(error)
            raise

    def finished(self, held_s: float = 0.0) -> None:
        """Give a place back, and let in whoever can go next."""
        self._counters.served_s += held_s
        self._scheduler.leave()

    def first_token(self, seconds: float, instance_id: str = "") -> None:
        """How long that request waited for its first token.

        Reported only for requests that asked for streaming — see `GatewayCounters`.
        Kept against the model that answered, because that is the only level at
        which the figure means anything.
        """
        counters = self._counters
        counters.first_token_s[instance_id] = (
            counters.first_token_s.get(instance_id, 0.0) + seconds)
        counters.first_tokens[instance_id] = (
            counters.first_tokens.get(instance_id, 0) + 1)

    def adopt_what_is_there(self) -> None:
        """Tell the scheduler what is already on the card, once after an outside change."""
        self._loader.adopt_what_is_there(self._scheduler.adopt)

    # -- what the scheduler asks of us --------------------------------------

    def _switch(self, shape: Shape, victims: tuple = ()) -> None:
        """Take those off and put this on: the loader's job, with the lock dropped."""
        self.replacements.switched(shape.instance_id, [victim.instance_id for victim in victims])
        self._loader.put_on_card(shape, victims)

    def _make_room(self, shape: Shape, loaded: list[dict]) -> list | None:
        """Which loaded shapes must go for this one, from the last memory reading."""
        return self._loader.make_room(shape, loaded)

    def _places(self, shape: Shape | None) -> int:
        """How many requests this shape serves at once. The engine's own number.

        One when it cannot be worked out — a shape naming an entry that has
        been deleted, say. Guessing high there would let requests through to a
        model that cannot take them.
        """
        if shape is None:
            return 1
        return ask_engine(self._operations, shape,
                          lambda engine, settings: max(1, engine.concurrency(settings)), 1)

    # -- how long to wait, per kind of request --------------------------

    def timeouts_for(self, task: Task) -> tuple[float, float]:
        """(first_byte_s, between_bytes_s) for this task.

        An override for `task.value` in `task_timeouts` wins; anything not
        named there gets the machine's default pair. Text generation is not
        given its own entry — it *is* the default, since every number above
        was sized against it — but a caller may still add one.
        """
        override = self.task_timeouts.get(task.value, {})
        return (float(override.get("first_byte_s", self.first_byte_s)),
                float(override.get("between_bytes_s", self.between_bytes_s)))

    # -- what the buttons on the page have to tell us -----------------------

    def apply_settings(self, settings: dict) -> None:
        """Take new limits without a restart.

        The queue length is the scheduler's; the two waits are used by the web
        layer when it forwards, and it reads them from here each time, so a
        change reaches the next request rather than the next restart.
        """
        if "first_byte_s" in settings:
            self.first_byte_s = float(settings["first_byte_s"])
        if "between_bytes_s" in settings:
            self.between_bytes_s = float(settings["between_bytes_s"])
        if "max_waiting" in settings:
            self._scheduler.max_waiting = int(settings["max_waiting"])
        if "task_timeouts" in settings:
            self.task_timeouts = dict(settings["task_timeouts"])
        patience = {key: settings[key] for key in self.patience.as_dict() if key in settings}
        if patience:
            self.patience = replace(self.patience, **patience)

    def engine_stalled(self, instance_id: str, reason: str) -> bool:
        """An engine had requests and did no work: restart it, once (`restart.py`)."""
        return self._watches.stalls.report(instance_id, reason)

    def _restart_stalled(self, instance_id: str, reason: str) -> None:
        """Stop the stalled engine and turn away whoever waits for it; the next request loads it."""
        try:
            self._operations.unload(instance_id)
        except Exception as error:
            self._counters.last_error = f"{reason}; stopping {instance_id} failed: {error}"
        else:
            self._counters.last_error = f"{instance_id} restarted: {reason}"
        self.reset(f"{instance_id} was restarted: {reason}")
        self._watches.book.forget(instance_id)

    def card_changed(self) -> None:
        """Something outside loaded or unloaded a model.

        The scheduler believes what it put there. A button on the page can put
        something else, and the next request would then be admitted onto a card
        that no longer holds what it thinks.
        """
        self._scheduler.forget_current()
        self._loader.card_changed()

    def reset(self, reason: str) -> int:
        """Throw everything away. What a forced stop means.

        Everything in flight is already dying — whoever forced it killed the
        engine. Everyone waiting is turned away rather than left queueing for a
        state that no longer exists. Returns how many were turned away.
        """
        self._loader.card_changed()
        self.replacements.stopped(reason, self._scheduler.state()["loaded"])
        return self._scheduler.reset(reason)

    # -- what the buttons on the page have to respect -----------------------

    def busy(self) -> dict | None:
        """Who holds the card now, for the Load and Unload buttons; None when idle."""
        return self.control.busy()

    def guard(self, action: str, instance_id: str) -> None:
        """Refuse a button that would interrupt an answer in progress."""
        self.control.guard(action, instance_id)

    # -- what the interface shows ------------------------------------------

    def stats(self) -> dict:
        """Everything the Gateway page shows, read now."""
        snapshot = GatewayStats(StatsSources(
            operations=self._operations, scheduler=self._scheduler, counters=self._counters,
            waits=(self.first_byte_s, self.between_bytes_s), adopt=self.adopt_what_is_there,
            holder=self.busy, engine_name_of=self.engine_name_of)).snapshot()
        return {**snapshot, **self.patience.as_dict()}
