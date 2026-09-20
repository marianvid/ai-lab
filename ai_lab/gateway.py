"""Model routing, admission and load coordination for agent requests.

Scheduling, memory accounting, request state and telemetry live in their
respective modules. See docs/gateway-behavior.md for the operating contract.
"""

from __future__ import annotations

import time

from . import budget
from .operations import Operations
from .eviction import EvictionPlanner
from .gateway_stats import GatewayCounters, GatewayStats
from .gateway_state import Lease, Shape
from .gateway_errors import CardBusy, CouldNotLoad, NotConfigured, ShapeNotServed
from .gateway_resources import GatewayResources
from .gateway_control import GatewayControl
from .scheduler import Abandoned, Scheduler, WillNotFit
from .types import Task


# How quiet the card has to be before a new model is loaded, and how long to
# wait for that. The threshold is not zero because a driver keeps a small
# allocation of its own; on this machine an idle card reads about 2 MiB.
QUIET_MB = 512.0
QUIET_TIMEOUT_S = 60.0
QUIET_POLL_S = 0.5

# How long to wait for an engine, and how many requests to hold. All three are
# defaults: the real values live in the configuration, because the right
# numbers differ between a card that reads 8,400 tokens of prompt in under a
# second and a Mac running a 70 GB model at 17 tokens a second.
FIRST_BYTE_S = 120.0
BETWEEN_BYTES_S = 30.0
MAX_WAITING = 150


class Gateway:
    """Routes by model name and puts that model on the card if it is not there.

    The queueing itself is `scheduler.Scheduler`, which knows nothing about
    models: it is handed a way to put a shape on the card and a way to ask how
    many requests that shape serves at once. Everything about names, engines
    and settings is here; everything about who goes next is there.
    """

    HISTORY = 50

    def __init__(self, operations: Operations,
                 quiet_mb: float = QUIET_MB,
                 quiet_timeout_s: float = QUIET_TIMEOUT_S,
                 poll_s: float = QUIET_POLL_S,
                 first_byte_s: float = FIRST_BYTE_S,
                 between_bytes_s: float = BETWEEN_BYTES_S,
                 max_waiting: int = MAX_WAITING,
                 task_timeouts: dict | None = None,
                 max_upload_bytes: int = 0,
                 max_upload_pixels: int = 0,
                 max_upload_dimension: int = 0) -> None:
        self.operations = operations
        self.quiet_mb = quiet_mb
        self.quiet_timeout_s = quiet_timeout_s
        self.poll_s = poll_s
        self.first_byte_s = first_byte_s
        self.between_bytes_s = between_bytes_s
        # Per-task overrides of the two waits above, keyed by `Task.value`.
        # A slow image workflow and a fast text completion share one gateway
        # but must not share one timeout: the default alone is sized for text.
        # Missing keys, or a task with no override, fall back to the defaults.
        self.task_timeouts = dict(task_timeouts or {})
        self.max_upload_bytes = max_upload_bytes
        self.max_upload_pixels = max_upload_pixels
        self.max_upload_dimension = max_upload_dimension
        # What the last card reading said, per pool. Refreshed around every
        # load, never from inside the scheduler's lock.
        self._budget_pools: dict = {}
        self.resources = GatewayResources(operations, quiet_mb,
                                          quiet_timeout_s, poll_s)
        self.eviction = EvictionPlanner()
        self.scheduler = Scheduler(self._put_on_card, self._places,
                                   self._make_room,
                                   max_waiting=max_waiting)
        # Whether anything outside may have changed what is on the card. Set at
        # startup and whenever a button on the page loads or unloads something.
        # The next switch sweeps: one expensive read after an outside change
        # rather than one on every request.
        self._resweep = True
        self.control = GatewayControl(self.scheduler)
        self.counters = GatewayCounters()

    # -- what a client can ask for -----------------------------------------

    def catalogue(self) -> list[dict]:
        """Every configured entry, loaded or not, with the names it answers to.

        Entries that are not loaded are listed too. A client is meant to be able
        to ask for one of them — that is the whole point.
        """
        return [{
            "id": instance["id"],
            "model_id": instance["model_id"],
            "engine": instance["engine"],
            "port": instance["port"],
            "loaded": bool(instance["running"]),
            "ready": bool(instance["ready"]),

            # Which shapes of request this one answers. A client that speaks
            # only one of them can tell from the listing which models are open
            # to it, instead of finding out by being refused.
            "shapes": self._shapes(instance),
        } for instance in self.operations.instances()]

    def _shapes(self, instance: dict) -> list[str]:
        try:
            task = Task(instance.get("task", Task.TEXT_GENERATION.value))
            return list(self.operations.engines.get(instance["engine"]).api_paths(task))
        except KeyError:
            return []

    @staticmethod
    def _engine_name(instance: dict) -> str:
        """What this entry's engine calls its own model.

        The last segment of the model path, which is the name the engine is
        started with and the only one it will answer to.
        """
        model_id = instance.get("model_id") or ""
        return model_id.rsplit("/", 1)[-1]

    def resolve(self, wanted: str, instances: list[dict] | None = None) -> dict:
        """The entry with this id, or NotConfigured naming what is known.

        The id and nothing else. It used to answer to four names — the id, the
        label a person gave it, the model's path, and the file at the end of
        that path — and the first match won.

        That is a collision waiting to be found. Two entries pointing at one
        model with different settings is not a strange thing to want; it is
        exactly what the settings in a request are for. Both would have
        answered to the file's name, one of them would have won silently, and
        the request would have been served by the wrong one.

        So there is one name. It is stable, it survives renaming the label, it
        has no spaces in it, and it fits the single `model` field the request
        shapes give us. The label is for reading; this is for sending.
        """
        key = (wanted or "").strip().lower()
        if instances is None:
            instances = self.operations.configured()
        for instance in instances:
            if instance["id"].strip().lower() == key:
                return instance
        known = sorted(instance["id"] for instance in instances)
        raise NotConfigured(
            f"No configured model answers to {wanted!r}. Known: {', '.join(known)}")

    # -- taking a place on the card -----------------------------------------

    def acquire(self, wanted: str, shape: str | None = None,
                settings: dict | None = None,
                still_wanted=None) -> Lease:
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

        `still_wanted()` says whether the client is still there. Asked when this
        request reaches the head of the queue and before anything is unloaded.
        A client that gave up must not cost a swap — reproduced on the machine
        before this existed: one hung up at once and the manager took a working
        model off the card to load twenty-one gigabytes for nobody.
        """
        started = time.perf_counter()
        # The configuration, not the supervisor. Which entry answers to a name,
        # which engine runs it, what settings it has — all of that is the file,
        # at 0.05 ms. Asking what every instance is *doing* costs 73 ms on the
        # container, and this path needs that only when something outside may
        # have changed the card.
        entries = self.operations.configured()
        instance = self.resolve(wanted, entries)
        if shape is not None:
            self._refuse_wrong_shape(instance, shape, entries)
        instance_id, port = instance["id"], instance["port"]
        # Always the full settings, never only what was asked for. A request
        # naming no settings and one naming exactly the configured ones are
        # asking for the same thing, and comparing partial dictionaries would
        # make them different — a reload for nothing, on every other request.
        asked_for = self.operations.effective_params(instance_id, settings or {})
        self._adopt_what_is_there()
        requested = Shape.of(instance_id, asked_for)
        # Some engines load weights only on their first inference. The card
        # reading taken when the server started is then too optimistic for the
        # next model. Refresh only for a switch; warm requests stay cheap.
        if not any(item["shape"] == requested
                   for item in self.scheduler.state()["loaded"]):
            self._read_the_card()

        try:
            self.scheduler.enter(requested,
                                 still_wanted=still_wanted)
        except WillNotFit as error:
            # The scheduler knows the ordering and nothing about memory, so the
            # numbers are attached here. A client that can read them can correct
            # itself — a smaller context, a smaller share of the card — without
            # anybody parsing an English sentence.
            error.detail = {"ai_lab": self._why_it_does_not_fit(
                instance_id, asked_for)}
            self.counters.last_error = str(error)
            raise
        except Exception as error:
            # A client that gave up is not a fault of this machine, so it does
            # not become the error the page shows.
            if not isinstance(error, Abandoned):
                self.counters.last_error = str(error)
            raise
        try:
            self.counters.requests += 1
            self.counters.waited_s += time.perf_counter() - started
            self.counters.arrivals.append((time.time(), instance_id))
            return Lease(self, instance_id, port, self._engine_name(instance),
                         started=time.perf_counter())
        except BaseException:
            # The place has been taken. Anything that goes wrong between there
            # and handing it to the caller has to give it back, or the card
            # loses a place with nobody using it.
            self.scheduler.leave()
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

    def _places(self, shape: "Shape | None") -> int:
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

    # -- deciding who has to go --------------------------------------------

    def _make_room(self, shape: "Shape", loaded: list[dict]) -> "list | None":
        """Choose victims from cached memory figures under the scheduler lock."""
        return self.eviction.choose(
            shape, loaded, needs_mb=self._needs_mb,
            free_mb=self._free_mb(), capacity_mb=self._capacity_mb())

    def _why_it_does_not_fit(self, instance_id: str, asked_for: dict) -> dict:
        """The numbers behind a refusal, for a client to act on.

        Everything here is a fact about this machine right now. Nothing is
        remembered between requests: what a model took last time is knowledge
        that belongs to whatever is making the requests, not to this manager.
        """
        shape = Shape.of(instance_id, asked_for)
        pool = (self._budget_pools.get(budget.CARD)
                or self._budget_pools.get(budget.MACHINE) or {})
        return {
            "model": instance_id,
            "engine": self._engine_name_of(shape),
            "needed_mb": round(self._needs_mb(shape)),
            "capacity_mb": round(self._capacity_mb()),
            "available_mb": round(self._free_mb()),
            "pool": pool.get("name", ""),
            "loaded": [item.instance_id for item in self._scheduled_shapes()],
            "asked": dict(asked_for),
        }

    def _needs_mb(self, shape: "Shape") -> float:
        return self.resources.needs_mb(shape, self._card_total_mb())

    def _card_total_mb(self) -> float:
        return self.resources.card_total_mb()

    def _capacity_mb(self) -> float:
        return self.resources.capacity_mb(self._budget_pools)

    def _free_mb(self) -> float:
        return self.resources.free_mb(self._budget_pools)

    def _read_the_card(self) -> None:
        # Never called while holding the scheduler lock. Keep the last good
        # reading if the host probe fails during a model transition.
        try:
            self._budget_pools = self.resources.read_pools()
        except Exception:
            pass

    def _put_on_card(self, shape: "Shape", victims: tuple = ()) -> None:
        """Take those off, wait for the card to go quiet, then load this.

        `victims` is what the scheduler decided has to go. Empty means this
        fits beside what is already there and nothing is disturbed.

        A manager restart or a machine boot can leave engines up that nothing
        here knows about. Those are cleared too, on the first switch after such
        a change — see `_resweep` — because loading into whatever is left is
        how a model that is known to fit fails to fit.
        """
        started = time.perf_counter()
        unloaded = self._clear(victims)
        self._resweep = False
        self._wait_until_quiet(unloaded)
        self._read_the_card()
        self._refuse_if_still_full(shape)

        operation = self.operations.load(shape.instance_id, shape.as_dict() or None)
        if not operation.ok:
            raise CouldNotLoad(
                operation.error or f"{shape.instance_id} would not start")

        self._read_the_card()
        took = time.perf_counter() - started
        self.counters.switches += 1
        if unloaded:
            self.counters.evictions += len(unloaded)
        self.counters.switch_s += took
        self.counters.history.append({
            "at": time.time(), "loaded": shape.instance_id, "unloaded": unloaded,
            "took_s": round(took, 1), "load_ms": operation.total_ms,
        })
        del self.counters.history[:-self.HISTORY]

    def _clear(self, victims: tuple = ()) -> list[str]:
        """Unload what was chosen, and anything nothing here knows about.

        The second half is the point of sweeping rather than unloading by name:
        a manager restart can leave an engine running that the scheduler never
        adopted, and its memory is real whether or not this knows about it.
        """
        wanted = {shape.instance_id for shape in victims}
        known = {item.instance_id for item in self._scheduled_shapes()}
        stopped = []
        for instance in self.operations.instances():
            if not instance["running"]:
                continue
            # The chosen victims, and anything running that the scheduler
            # does not believe in. A stray is not in the budget arithmetic, so
            # its memory is unaccounted — and loading into what is left is how
            # a model that is known to fit fails to.
            if instance["id"] in wanted or instance["id"] not in known:
                self.operations.unload(instance["id"])
                stopped.append(instance["id"])
        return stopped

    def _scheduled_shapes(self) -> list:
        """What the scheduler believes is loaded."""
        return [item["shape"] for item in self.scheduler.state()["loaded"]]

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

    # -- which shapes an entry answers --------------------------------------

    def _answers(self, instance: dict, shape: str) -> bool:
        """Whether this entry's engine answers this kind of request."""
        try:
            engine = self.operations.engines.get(instance["engine"])
        except KeyError:
            return False
        task = Task(instance.get("task", Task.TEXT_GENERATION.value))
        return shape in engine.api_paths(task)

    def _refuse_wrong_shape(self, instance: dict, shape: str,
                            instances: list[dict]) -> None:
        """Say no, and say which entries would have worked.

        A client that sent the wrong shape does not know which of its models
        are on which engine, and has no way to find out from a refusal that
        only says no. Naming them turns a dead end into one edit.
        """
        if self._answers(instance, shape):
            return
        able = sorted(other["id"] for other in instances
                      if self._answers(other, shape))
        raise ShapeNotServed(
            f"{instance['id']} runs on {instance['engine']}, which does not "
            f"answer {shape}. "
            + (f"Configured models that do: {', '.join(able)}." if able
               else "No configured model answers it."))

    # -- what the buttons on the page have to respect -----------------------

    def busy(self) -> dict | None:
        return self.control.busy()

    def guard(self, action: str, instance_id: str) -> None:
        self.control.guard(action, instance_id)

    def _wait_until_quiet(self, stopped: list) -> None:
        self.resources._wait_until_quiet(stopped)

    def _wait_for_release(self, deadline: float, keeping: bool) -> None:
        self.resources._wait_for_release(deadline, keeping)

    def _refuse_if_still_full(self, shape: "Shape") -> None:
        """Say so before starting an engine that cannot fit.

        Asked only when the arithmetic is available. Where it is not, the
        machine was emptied instead and `_wait_for_release` has already
        insisted the card reached nothing.
        """
        needed = self._needs_mb(shape)
        free = self._free_mb()
        if not needed or not free or needed <= free:
            return
        raise CouldNotLoad(
            f"{shape.instance_id} needs about {needed / 1024:.1f} GB and "
            f"{free / 1024:.1f} GB is free after unloading what could be "
            f"unloaded. Something outside AI-Lab is holding the card, or an "
            f"engine did not release its memory.")

    def _loaded(self) -> str | None:
        """What is on the card, read rather than remembered.

        A remembered value goes stale the moment something outside the gateway
        unloads a model — and that happens on purpose: a person can force past
        a busy card from the page, and the gateway is deliberately not told.
        Reading it costs one call the caller is making anyway, and it cannot be
        wrong.

        One name, and the first one found. This is used where a single answer
        is wanted — the page asks the scheduler for the whole set. Anything
        running that the scheduler does not know about is swept off by the next
        switch, because its memory is real and unaccounted for.
        """
        for instance in self.operations.instances():
            if instance["running"]:
                return instance["id"]
        return None

    # -- what the interface shows ------------------------------------------

    def stats(self) -> dict:
        return GatewayStats(self).snapshot()

    def _engine_name_of(self, shape) -> str:
        if shape is None:
            return ""
        try:
            entry = self.operations.instance(shape.instance_id)
            return self.operations.engines.get(entry["engine"]).display_name
        except Exception:
            return ""

    def _shapes_offered(self) -> list[dict]:
        return GatewayStats(self)._shapes_offered()
