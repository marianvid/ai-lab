"""One address for an agent workflow that uses several models.

An agent asks the researcher model to read, the developer model to write, the
reviewer model to check. Each is a separate entry here on its own port, and only
one of them can be on the card. An agent that names a model which is not running
gets a refused connection, and the workflow stops there.

This removes that. A request names a model; if it is already loaded the request
goes through, and if it is not, the card is emptied and that model is loaded
first. The agent waits longer for that one request and sees nothing else.

## One model on the card. Many requests to it.

The card holds one model. That is the machine, not a decision.

Requests to the model that is on it are a different matter, and they run
together — up to the number the engine was started to serve. vLLM interleaves
them in the same pass, which is where its throughput comes from: measured on
this machine at up to seventeen times as concurrency rises, against about 1.4
for llama.cpp. Making them take turns threw that away.

The number is the engine's own, per entry: slots for llama.cpp, sequences for
vLLM. Ask the engine, do not guess from a setting name.

Requests wanting a different model wait, and the queueing rules are in
`scheduler.py` — including the two that are easy to get wrong and impossible to
notice: a busy model must stop admitting the moment somebody waits for another,
and a model just loaded must not admit arrivals ahead of who was already
waiting. Without either, one model starves the other for ever.

**The oldest waiting request decides what is loaded next.** Strictly, with no
trade for fewer loads — because the fifty requests for one model may be waiting
on the answer to the one request for another.

Which leads to the one thing to design workflows around: **a request must not
wait, inside itself, on another request to this same gateway.** Fill every
place with things that cannot finish and nothing finishes.

A model plus the settings it was started with is one "shape". Two requests for
the same model wanting different context sizes are not requests for the same
thing: one of them needs a reload.

## The buttons on the page are the other way in

Requests here are safe from each other: they queue. The Load and Unload buttons
are not part of that. They reach the engines directly and know nothing about
who is mid-answer, so pressing Unload during a long answer would kill it in the
middle of a sentence.

`guard` is what the routes behind those buttons call first. It refuses while
anything is running or waiting, and says what it found, so the page can offer
to go ahead anyway. Going ahead means a clean slate: everything in flight dies,
everyone waiting is turned away, the card is empty. Half-forced is worse than
either.

## Emptying the card properly

Switching does not unload the outgoing model and start the next one straight
away. The driver returns VRAM a moment after a process exits, and starting a
model on top of memory that has not come back yet fails in a way that reads like
the new model being too large.

So a switch unloads everything running, waits for the card to actually go quiet,
and only then loads. If it does not go quiet, the switch fails and says so,
rather than loading into a mess.

## How long to wait for an engine

Two limits, and they are limits of safety rather than of patience: in normal
work nothing comes near them.

**To the first byte** covers connecting and reading the prompt — and, for a
request that did not ask for streaming, the whole answer, because such an
engine sends nothing until it has finished. Measured here: 8,400 tokens of
prompt read in 0.78 s on the card. On a model split between card and system
memory, or on Apple silicon, it is far slower — that is what this number is
sized for.

**Between bytes** catches an engine that starts answering and stops. At the
slowest generation measured on either machine, 17 tokens a second, the gap
between them is 59 milliseconds. Anything of the order of seconds means
something is wrong, not slow.

One number cannot do both jobs: their sane values are four orders of magnitude
apart. The single hour-long timeout this replaced was absurd for one and
useless for the other.

There is no HTTP in this file. It decides which entry serves a name and makes
sure it is the one on the card; forwarding the request is the web layer's job.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from .operations import Operations
from .scheduler import Abandoned, Scheduler


class NotConfigured(KeyError):
    """No entry serves that model name.

    A KeyError so the web layer answers 404 without being told, since that is
    already the rule for "no such thing".
    """

    def __str__(self) -> str:
        # KeyError renders its argument with repr(), which wraps the whole
        # sentence in quotes and escapes what is inside it. This message lists
        # the names that would have worked, and it is read by a person
        # debugging an agent, so it should arrive as a sentence.
        return self.args[0] if self.args else ""


class CouldNotLoad(RuntimeError):
    """The entry exists but the card could not be made ready for it."""


class ShapeNotServed(ValueError):
    """The entry exists, but its engine does not answer that kind of request.

    A request can arrive in more than one shape, and not every engine speaks
    every shape. Refused here, with the entries that would have worked, rather
    than forwarded to an engine that would answer 404 about a path the client
    never chose.
    """


class CardBusy(RuntimeError):
    """The card is serving a request, and the action asked for would cut it off.

    Raised at the request of the interface, not by the gateway's own work: the
    buttons on the page reach the engines directly, and this is how they find
    out that somebody is mid-answer. It carries `detail` so the page can offer
    to go ahead anyway rather than only printing a sentence.
    """

    def __init__(self, message: str, holder: dict) -> None:
        super().__init__(message)
        self.detail = {"busy": holder}


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


@dataclass(slots=True)
class Lease:
    """A place on the card, held for the length of one request.

    Taken before the request is forwarded and given back after the last byte of
    the answer, including a streamed answer that takes a minute.

    Each lease knows whether it has been given back, because several are held
    at once now. A request that fails while being forwarded hands its place
    back twice — once from the code that noticed, once from the reader's
    cleanup — and without this the second would be taking a place from somebody
    else.
    """

    gateway: "Gateway"
    instance_id: str
    port: int
    # What the engine calls its own model. llama.cpp and vLLM are both started
    # with an explicit name and both refuse a request naming anything else, so
    # the name the client used has to be translated before forwarding. It is
    # carried here because it was known when the lease was made: asking for it
    # afterwards meant reading every instance's state a second time, which is
    # the expensive question, for an answer that is pure configuration.
    model_name: str = ""
    _given_back: bool = False

    def release(self) -> None:
        if self._given_back:
            return
        self._given_back = True
        self.gateway.finished()

    def __enter__(self) -> "Lease":
        return self

    def __exit__(self, *_exception) -> None:
        self.release()


@dataclass(frozen=True, slots=True)
class Shape:
    """A model, and the settings it has to be started with.

    Two requests for the same entry wanting different context sizes are not
    requests for the same thing — one of them needs a reload — so the settings
    are part of what is being asked for, not a note attached to it. Frozen and
    hashable, so equality is the whole test.
    """

    instance_id: str
    settings: tuple = ()                # (key, value) pairs, sorted

    @classmethod
    def of(cls, instance_id: str, settings: dict | None) -> "Shape":
        return cls(instance_id, tuple(sorted((settings or {}).items())))

    def as_dict(self) -> dict:
        return dict(self.settings)


@dataclass(slots=True)
class _Counters:
    requests: int = 0
    switches: int = 0
    waited_s: float = 0.0
    switch_s: float = 0.0
    last_error: str = ""
    history: list[dict] = field(default_factory=list)


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
                 max_waiting: int = MAX_WAITING) -> None:
        self.operations = operations
        self.quiet_mb = quiet_mb
        self.quiet_timeout_s = quiet_timeout_s
        self.poll_s = poll_s
        self.first_byte_s = first_byte_s
        self.between_bytes_s = between_bytes_s
        self.scheduler = Scheduler(self._put_on_card, self._places,
                                   max_waiting=max_waiting)
        # Whether anything outside may have changed what is on the card. Set at
        # startup and whenever a button on the page loads or unloads something.
        # The next switch sweeps: one expensive read after an outside change
        # rather than one on every request.
        self._resweep = True
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
            # Which shapes of request this one answers. A client that speaks
            # only one of them can tell from the listing which models are open
            # to it, instead of finding out by being refused.
            "shapes": self._shapes(instance),
        } for instance in self.operations.instances()]

    def _shapes(self, instance: dict) -> list[str]:
        try:
            return list(self.operations.engines.get(instance["engine"]).api_paths())
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

    def resolve(self, wanted: str, instances: list[dict] | None = None) -> dict:
        """The entry serving this name, or NotConfigured naming what is known.

        Takes an optional list already read, because reading it is not cheap:
        it asks the supervisor about every configured instance and probes each
        one that is up. See `acquire`, which reads it once and passes it on.
        """
        key = (wanted or "").strip().lower()
        if instances is None:
            instances = self.operations.instances()
        for instance in instances:
            if any(alias.lower() == key for alias in self._aliases(instance)):
                return instance
        known = sorted({alias for instance in instances
                        for alias in self._aliases(instance)})
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
        instances = self.operations.instances()
        instance = self.resolve(wanted, instances)
        if shape is not None:
            self._refuse_wrong_shape(instance, shape, instances)
        instance_id, port = instance["id"], instance["port"]
        # Always the full settings, never only what was asked for. A request
        # naming no settings and one naming exactly the configured ones are
        # asking for the same thing, and comparing partial dictionaries would
        # make them different — a reload for nothing, on every other request.
        asked_for = self.operations.effective_params(instance_id, settings or {})
        self._adopt_what_is_there(instances)

        try:
            self.scheduler.enter(Shape.of(instance_id, asked_for),
                                 still_wanted=still_wanted)
        except Exception as error:
            if not isinstance(error, Abandoned):
                self.counters.last_error = str(error)
            raise
        self.counters.requests += 1
        self.counters.waited_s += time.perf_counter() - started
        return Lease(self, instance_id, port, self._engine_name(instance))

    def finished(self) -> None:
        """Give a place back, and let in whoever can go next."""
        self.scheduler.leave()

    def _adopt_what_is_there(self, instances: list[dict]) -> None:
        """Tell the scheduler what is already on the card, once.

        Only after something outside may have changed it — a manager that has
        just started, or a button on the page. Otherwise the scheduler is the
        authority and reading again would only be a chance to disagree with
        itself.

        Exactly one model, and answering: anything else is left as unknown, and
        the next request switches, which unloads whatever is there. Two engines
        up is not a card to adopt, it is a card to clear.
        """
        if not self._resweep:
            return
        self._resweep = False
        running = [item for item in instances if item["running"] and item["ready"]]
        if len(running) != 1 or any(item["running"] for item in instances
                                    if item["id"] != running[0]["id"]):
            return
        found = running[0]
        self.scheduler.adopt(Shape.of(
            found["id"], {**found.get("params", {}),
                          **found.get("active_params", {})}))

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

    def _put_on_card(self, shape: "Shape") -> None:
        """Empty the card, wait for it to go quiet, then load.

        Everything running is unloaded, not only what was last asked for: a
        manager restart or a machine boot can leave more than one engine up,
        and loading into whatever is left is how a curated model that is known
        to fit fails to fit.
        """
        started = time.perf_counter()
        unloaded = self._clear()
        self._resweep = False
        self._wait_until_quiet()

        operation = self.operations.load(shape.instance_id, shape.as_dict() or None)
        if not operation.ok:
            raise CouldNotLoad(
                operation.error or f"{shape.instance_id} would not start")

        took = time.perf_counter() - started
        self.counters.switches += 1
        self.counters.switch_s += took
        self.counters.history.append({
            "at": time.time(), "loaded": shape.instance_id, "unloaded": unloaded,
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
        return stopped

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
        return shape in engine.api_paths()

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
        """What is on the card and what it is doing, or None if it is idle.

        Idle means nothing running and nobody waiting. A queue with nothing in
        flight still counts as busy: a switch is about to happen, and stopping
        a model in that moment is as disruptive as stopping one mid-answer.
        """
        state = self.scheduler.state()
        if not state["in_flight"] and not state["waiting"] and not state["switching"]:
            return None
        current = state["current"]
        return {
            "instance_id": current.instance_id if current else "",
            "answering": bool(state["in_flight"]),
            "in_flight": state["in_flight"],
            "places": state["places"],
            "waiting": len(state["waiting"]),
            "switching": state["switching"],
        }

    def guard(self, action: str, instance_id: str) -> None:
        """Refuse an action that would interrupt work in progress.

        Requests here are safe from each other: they queue. This is for the
        other way in — the Load and Unload buttons reach the engines directly
        and know nothing about leases or queues. Without this, pressing Unload
        during a long answer kills it mid sentence, and the agent sees a
        connection that simply stopped.

        It names what it found, because "busy" is not enough to decide with:
        one answer being written is a different thing from forty requests
        waiting for a model to load.
        """
        holder = self.busy()
        if holder is None:
            return
        who = holder["instance_id"] or "a model"
        parts = []
        if holder["in_flight"]:
            answers = holder["in_flight"]
            parts.append(f"{who} is answering "
                         f"{answers} request{'' if answers == 1 else 's'}")
        elif holder["switching"]:
            parts.append(f"{who} is being loaded")
        if holder["waiting"]:
            waiting = holder["waiting"]
            parts.append(f"{waiting} more {'is' if waiting == 1 else 'are'} waiting")
        raise CardBusy(
            f"{' and '.join(parts) or 'The card is in use'}. Going ahead with "
            f"'{action}' on {instance_id} cuts all of that off.", holder)

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

    def _loaded(self) -> str | None:
        """What is on the card, read rather than remembered.

        A remembered value goes stale the moment something outside the gateway
        unloads a model — and that happens on purpose: a person can force past
        a busy card from the page, and the gateway is deliberately not told.
        Reading it costs one call the caller is making anyway, and it cannot be
        wrong.

        One name, not a list, because one model on the card is the rule. If two
        are somehow up, the next request through here unloads the stray.
        """
        for instance in self.operations.instances():
            if instance["running"]:
                return instance["id"]
        return None

    # -- what the interface shows ------------------------------------------

    def stats(self) -> dict:
        """Enough to see what is happening without reading a log.

        Two numbers matter and they answer different questions.

        **Switches as a share of requests** says whether the workflow is
        working or loading. Close to one means it changes model on almost every
        step, and each change is an unload, a wait and a load.

        **Waiting** says whether requests are stuck behind each other. That is
        a different fault with a different fix: too little concurrency, or two
        models fighting over one card.
        """
        counters = self.counters
        state = self.scheduler.state()
        current = state["current"]
        waiting = state["waiting"]
        return {
            "current": current.instance_id if current else None,
            "current_settings": current.as_dict() if current else {},
            "busy": bool(state["in_flight"] or waiting or state["switching"]),
            "holder": self.busy(),
            "in_flight": state["in_flight"],
            "places": state["places"],
            "switching": state["switching"],
            # What is waiting, and for how long. A request that has been in the
            # queue for a minute is a fact worth seeing before it becomes a
            # complaint.
            "waiting": len(waiting),
            "waiting_for": _waiting_summary(waiting),
            "longest_wait_s": max((item["waiting_s"] for item in waiting),
                                  default=0.0),
            "max_waiting": self.scheduler.max_waiting,
            "first_byte_s": self.first_byte_s,
            "between_bytes_s": self.between_bytes_s,
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


def _waiting_summary(waiting: list[dict]) -> list[dict]:
    """How many are waiting for each model, oldest first.

    By model rather than by shape: somebody reading the page wants to know
    which models are contended, and two context sizes of one model read as one
    queue to them.
    """
    grouped: dict[str, dict] = {}
    for item in waiting:
        shape = item["shape"]
        name = getattr(shape, "instance_id", str(shape))
        row = grouped.setdefault(name, {"instance_id": name, "waiting": 0,
                                        "longest_wait_s": 0.0})
        row["waiting"] += 1
        row["longest_wait_s"] = max(row["longest_wait_s"], item["waiting_s"])
    return sorted(grouped.values(), key=lambda row: -row["longest_wait_s"])
