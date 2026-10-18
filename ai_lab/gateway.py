"""Model routing, admission and load coordination for agent requests.

Scheduling, memory accounting, request state and telemetry live in their
respective modules; the listing is `gateway_catalogue`, making room and loading
is `gateway_loading` (both base classes of `Gateway`). See
docs/gateway-behavior.md for the operating contract.
"""

from __future__ import annotations

import time

from .eviction import EvictionPlanner
from .gateway_catalogue import GatewayCatalogue
from .gateway_control import GatewayControl
from .gateway_errors import CardBusyError, CouldNotLoadError, NotConfiguredError, ShapeNotServedError
from .gateway_limits import GatewayLimits
from .gateway_loading import CardLoading
from .gateway_resources import GatewayResources
from .gateway_state import Lease, Shape
from .gateway_stats import GatewayCounters, GatewayStats
from .operations import Operations
from .scheduler import AbandonedError, Scheduler, WillNotFitError
from .scheduler_parts import StillWanted
from .types import Task

# The gateway's errors are part of its interface: callers import them from here.
__all__ = ["CardBusyError", "CouldNotLoadError", "Gateway", "GatewayLimits", "NotConfiguredError",
           "ShapeNotServedError"]


class Gateway(GatewayCatalogue, CardLoading):
    """Routes by model name and puts that model on the card if it is not there.

    The queueing itself is `scheduler.Scheduler`, which knows nothing about
    models: it is handed a way to put a shape on the card and a way to ask how
    many requests that shape serves at once. Everything about names, engines
    and settings is here; everything about who goes next is there.
    """

    def __init__(self, operations: Operations, limits: GatewayLimits | None = None) -> None:
        limits = limits or GatewayLimits()
        self.operations = operations
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
        # What the last card reading said, per pool. Refreshed around every
        # load, never from inside the scheduler's lock.
        self._budget_pools: dict = {}
        self.resources = GatewayResources(operations, limits.quiet_mb,
                                          limits.quiet_timeout_s, limits.poll_s)
        self.eviction = EvictionPlanner()
        self.scheduler = Scheduler(self._put_on_card, self._places, self._make_room,
                                   max_waiting=limits.max_waiting)
        # Whether anything outside may have changed what is on the card. Set at
        # startup and whenever a button on the page loads or unloads something.
        # The next switch sweeps: one expensive read after an outside change
        # rather than one on every request.
        self._resweep = True
        self.control = GatewayControl(self.scheduler)
        self.counters = GatewayCounters()

    # -- taking a place on the card -----------------------------------------

    def acquire(self, wanted: str, shape: str | None = None,
                settings: dict | None = None,
                still_wanted: StillWanted | None = None) -> Lease:
        """Take a place on the card, with the named model on it.

        Returns once the model is answering and this request may go through.
        It may return at once — the model is loaded and there is room — or after
        waiting for a place, or after waiting for a switch and the load.

        `shape` is the kind of request about to be forwarded. Checked before
        anything is queued: an entry whose engine does not answer that shape is
        refused straight away rather than after a forty-second load.

        `settings` asks for the model started with something other than what the
        entry is configured with. Checked before queueing too, so a setting the
        engine does not have comes back at once. It is part of what is being
        asked for: two requests wanting different context sizes cannot share a
        card, whatever the model.

        `still_wanted()` says whether the request is still to be served, asked
        at the head of the queue before anything is unloaded: a client that
        gave up must not cost a swap. It may also withdraw the request with a
        reason (`AdmissionWithdrawnError`) — a reservation granted meanwhile
        (ADR 0005).
        """
        started = time.perf_counter()
        instance, asked_for = self._what_is_asked(wanted, shape, settings)
        self._enter(Shape.of(instance["id"], asked_for), asked_for, still_wanted)
        try:
            self.counters.requests += 1
            self.counters.waited_s += time.perf_counter() - started
            self.counters.arrivals.append((time.time(), instance["id"]))
            return Lease(self, instance["id"], instance["port"], self._engine_name(instance),
                         started=time.perf_counter())
        except BaseException:
            # The place has been taken. Anything that goes wrong between there
            # and handing it to the caller has to give it back, or the card
            # loses a place with nobody using it.
            self.scheduler.leave()
            raise

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
        entries = self.operations.configured()
        instance = self.resolve(wanted, entries)
        if shape is not None:
            self._refuse_wrong_shape(instance, shape, entries)
        return instance, self.operations.effective_params(instance["id"], settings or {})

    def _enter(self, requested: Shape, asked_for: dict,
               still_wanted: StillWanted | None) -> None:
        """Wait in the scheduler's queue until this shape has a place."""
        self._adopt_what_is_there()
        # Some engines load weights only on their first inference. The card
        # reading taken when the server started is then too optimistic for the
        # next model. Refresh only for a switch; warm requests stay cheap.
        if not any(item["shape"] == requested
                   for item in self.scheduler.state()["loaded"]):
            self._read_the_card()
        try:
            self.scheduler.enter(requested, still_wanted=still_wanted)
        except WillNotFitError as error:
            # The scheduler knows the ordering and nothing about memory, so the
            # numbers are attached here. A client that can read them can correct
            # itself — a smaller context, a smaller share of the card — without
            # anybody parsing an English sentence.
            error.detail = {"ai_lab": self._why_it_does_not_fit(
                requested.instance_id, asked_for)}
            self.counters.last_error = str(error)
            raise
        except Exception as error:
            # A client that gave up, or a request withdrawn by its reservation,
            # is not a fault of this machine, so it does not become the error
            # the page shows.
            if not isinstance(error, AbandonedError):
                self.counters.last_error = str(error)
            raise

    def finished(self, held_s: float = 0.0) -> None:
        """Give a place back, and let in whoever can go next."""
        self.counters.served_s += held_s
        self.scheduler.leave()

    def first_token(self, seconds: float, instance_id: str = "") -> None:
        """How long that request waited for its first token.

        Reported only for requests that asked for streaming — see `GatewayCounters`.
        Kept against the model that answered, because that is the only level at
        which the figure means anything.
        """
        counters = self.counters
        counters.first_token_s[instance_id] = (
            counters.first_token_s.get(instance_id, 0.0) + seconds)
        counters.first_tokens[instance_id] = (
            counters.first_tokens.get(instance_id, 0) + 1)

    def _adopt_what_is_there(self) -> None:
        """Tell the scheduler what is already on the card, once.

        Only after something outside may have changed it — a manager that has
        just started, or a button on the page. Otherwise the scheduler is the
        authority and reading again would only be a chance to disagree with
        itself.

        Everything that is up and answering, however many. This used to insist
        on exactly one and give up otherwise — two engines up was not a card to
        adopt but a card to clear — and with a memory budget that is simply
        the ordinary state. The page reported "nothing loaded" while two models
        were serving.

        A model that is up but not yet answering is left for later, and the
        flag stays set so the question is asked again. Adopting it would send
        requests to a port with nothing behind it; giving up on it for good
        would leave it invisible until something else changed.
        """
        if not self._resweep:
            return
        # The expensive question, asked once after an outside change rather
        # than on every request.
        instances = self.operations.instances()
        running = [item for item in instances if item["running"]]
        if any(not item["ready"] for item in running):
            return                      # still coming up: ask again next time
        self._resweep = False
        self.scheduler.adopt(*[
            Shape.of(item["id"], {**item.get("params", {}),
                                  **item.get("active_params", {})})
            for item in running])

    # -- what the scheduler asks of us --------------------------------------

    def _places(self, shape: Shape | None) -> int:
        """How many requests this shape serves at once. The engine's own number.

        One when it cannot be worked out — a shape naming an entry that has
        been deleted, say. Guessing high there would let requests through to a
        model that cannot take them.
        """
        if shape is None:
            return 1
        try:
            instance = self.operations.instance(shape.instance_id)
            engine = self.operations.engines.get(instance["engine"])
            return max(1, engine.concurrency({**instance["params"],
                                              **shape.as_dict()}))
        except Exception:
            return 1

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
            self.scheduler.max_waiting = int(settings["max_waiting"])
        if "task_timeouts" in settings:
            self.task_timeouts = dict(settings["task_timeouts"])

    def card_changed(self) -> None:
        """Something outside loaded or unloaded a model.

        The scheduler believes what it put there. A button on the page can put
        something else, and the next request would then be admitted onto a card
        that no longer holds what it thinks.
        """
        self.scheduler.forget_current()
        self._resweep = True

    def reset(self, reason: str) -> int:
        """Throw everything away. What a forced stop means.

        Everything in flight is already dying — whoever forced it killed the
        engine. Everyone waiting is turned away rather than left queueing for a
        state that no longer exists. Returns how many were turned away.
        """
        self._resweep = True
        return self.scheduler.reset(reason)

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
        return GatewayStats(self).snapshot()

    def _shapes_offered(self) -> list[dict]:
        return GatewayStats(self)._shapes_offered()
