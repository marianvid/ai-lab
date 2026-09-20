"""Read-only reporting for gateway state, queues, load, and supported routes.

The gateway owns admission and switching; this object projects that state for
people and agents without adding another set of scheduling rules.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

from . import budget
from .types import Task


@dataclass(slots=True)
class GatewayCounters:
    """What the page reports, and nothing that only ever goes up.

    A lifetime total of requests says nothing: it grows while you watch it and
    means the same at 40 as at 40,000. What is worth showing is a rate, an
    average, and a share — figures that stay comparable to themselves.
    """

    requests: int = 0
    switches: int = 0
    # How many models were pushed off to make room. The number that hurts:
    # a load beside what is there costs a load; a load that displaces
    # something costs that too, and the next request for it.
    evictions: int = 0
    waited_s: float = 0.0
    switch_s: float = 0.0
    # Time spent actually answering, summed. The denominator for "how much of
    # the working time went on loading models" — the wall clock is no use
    # there, because a machine that sits idle overnight would report a
    # flattering number for a workflow that spends its life swapping.
    served_s: float = 0.0
    # When each request arrived and which model it wanted, for a rate rather
    # than a total. Pruned to the last minute whenever it is read.
    #
    # The model is kept because the total answers "how busy is this machine"
    # and the split answers "which model is carrying it" — and the second is
    # what decides which one is worth keeping loaded.
    arrivals: deque = field(default_factory=lambda: deque(maxlen=4096))
    # Time to the first token, over requests that asked for streaming. Only
    # those: without streaming an engine sends nothing until the answer is
    # finished, so its "first byte" is the whole generation and averaging the
    # two together measures neither.
    #
    # Kept per model and never totalled. A 3B and a 35B have first-token times
    # that differ by an order of magnitude, and one average across both is a
    # figure that describes neither. With one model on the machine the average
    # was right by accident.
    first_token_s: dict = field(default_factory=dict)
    first_tokens: dict = field(default_factory=dict)
    last_error: str = ""
    history: list[dict] = field(default_factory=list)


class GatewayStats:
    def __init__(self, gateway):
        self.gateway = gateway
        self.operations = gateway.operations
        self.scheduler = gateway.scheduler
        self.counters = gateway.counters
        self.first_byte_s = gateway.first_byte_s
        self.between_bytes_s = gateway.between_bytes_s

    # -- what the interface shows ------------------------------------------

    def snapshot(self) -> dict:
        """Enough to see what is happening without reading a log.

        Two numbers matter and they answer different questions.

        **Switches as a share of requests** says whether the workflow is
        working or loading. Close to one means it changes model on almost every
        step, and each change is an unload, a wait and a load.

        **Waiting** says whether requests are stuck behind each other. That is
        a different fault with a different fix: too little concurrency, or two
        models fighting over one card.
        """
        # The page is as good a reason to look as a request is. Without this
        # a manager that has just started reports an empty card until somebody
        # sends something — which is exactly when a person is watching it.
        # `_adopt_what_is_there` clears the flag, so this costs the expensive
        # read once after a restart rather than on every refresh.
        self.gateway._adopt_what_is_there()
        counters = self.counters
        state = self.scheduler.state()
        waiting = state["waiting"]
        return {
            # A list, always, even while the scheduler still holds exactly one.
            # What is on the machine is a set — one model today, more when the
            # budget allows it — and a report shaped as a single name would
            # have to change shape later, taking every reader with it. The
            # engine is beside each name because one tells you what to send and
            # the other tells you what will answer, and the second decides
            # which request shapes work.
            "loaded": [{
                "instance_id": item["shape"].instance_id,
                "engine": self.gateway._engine_name_of(item["shape"]),
                "settings": item["shape"].as_dict(),
                "in_flight": item["in_flight"],
                "places": item["places"],
                # How many are queued for this one specifically. With several
                # loaded, a request can be waiting because its own model is
                # full rather than because a swap is coming.
                "waiting": item["waiting"],
                # Per model, because neither figure means anything averaged
                # across two: a 3B and a 35B differ by an order of magnitude on
                # the first, and the second answers "which of these is carrying
                # the traffic", which is what decides what is worth keeping.
                "requests_per_minute": self._rate_of(item["shape"].instance_id),
                "first_token_s": self._first_token_of(item["shape"].instance_id),
            } for item in state["loaded"]],
            "busy": bool(state["in_flight"] or waiting or state["switching"]),
            "holder": self.gateway.busy(),
            "in_flight": state["in_flight"],
            "switching": state["switching"],
            # What is waiting, and for how long. A request that has been in the
            # queue for a minute is a fact worth seeing before it becomes a
            # complaint.
            "waiting": len(waiting),
            "waiting_for": _waiting_summary(waiting),
            "queue_runs": _runs(waiting),
            "longest_wait_s": max((item["waiting_s"] for item in waiting),
                                  default=0.0),
            "max_waiting": self.scheduler.max_waiting,
            "first_byte_s": self.first_byte_s,
            "between_bytes_s": self.between_bytes_s,
            # Which shapes of request each model answers. Configuration only
            # — an entry's engine and what that engine serves — so the page can
            # say it without the expensive question.
            "shapes": self._shapes_offered(),
            # What the machine has room for, pool by pool. The same answer an
            # admission decision would use, so the page cannot disagree with
            # the thing that says no. One accelerator reading gives this and
            # the temperature both — 35 ms on the container, against a page
            # that is otherwise 9.
            **self._memory_and_heat(),
            "requests_per_minute": self._rate(),
            "switches": counters.switches,
            "evictions": counters.evictions,
            # Of the time this was working — answering or loading — how much
            # went on loading. Against the wall clock instead, a machine that
            # sits idle overnight reports a flattering number for a workflow
            # that spends its life swapping.
            "switching_share": self._switching_share(),
            "average_wait_s": round(counters.waited_s / counters.requests, 2)
                              if counters.requests else 0.0,
            "last_error": counters.last_error,
            "recent": list(reversed(counters.history[-10:])),
        }


    def _memory_and_heat(self) -> dict:
        """What the machine has room for, and how warm the card is.

        One accelerator reading answers both. On Linux each is an `nvidia-smi`
        — 30 ms against a page that is otherwise 9 — and this used to take two
        of them, which also let the two halves disagree by however long passed
        between.

        The memory comes through `budget` rather than being worked out here, so
        that this page and the thing that refuses a model give the same answer.
        Two figures meaning the same and computed twice eventually differ, and
        the one on screen is the one nobody checks.

        Utilisation is in the same reading and is not reported: it is an
        instantaneous sample, so a five-second page lands between requests more
        often than not and shows nought per cent on a machine working steadily.
        A figure that is usually wrong is worse than none.
        """
        try:
            card = self.operations.host.accelerator()
        except Exception:
            return {"memory": {}, "card": {}}
        try:
            found = budget.of(self.operations.host, self.operations.reserve_mb(),
                              card=card)
            memory = found.json()
        except Exception:
            memory = {}
        # Unified memory has no temperature to read, and that comes back empty
        # rather than as a number meaning something else.
        return {"memory": memory, "card": {"temperature_c": card.temperature_c}}

    def _rate(self) -> float:
        """Requests in the last minute. Zero when nothing is happening.

        A rate rather than a total: a lifetime count grows while you watch it
        and means the same at 40 as at 40,000.
        """
        return len(self._recent_arrivals())

    def _rate_of(self, instance_id: str) -> float:
        """Requests in the last minute for one model."""
        return sum(1 for _, wanted in self._recent_arrivals()
                   if wanted == instance_id)

    def _recent_arrivals(self) -> deque:
        """The last minute of arrivals, pruning what has aged out."""
        arrivals = self.counters.arrivals
        cutoff = time.time() - 60.0
        while arrivals and arrivals[0][0] < cutoff:
            arrivals.popleft()
        return arrivals

    def _first_token_of(self, instance_id: str) -> float:
        """Average time to the first token for one model, over streamed requests."""
        many = self.counters.first_tokens.get(instance_id, 0)
        if not many:
            return 0.0
        return round(self.counters.first_token_s.get(instance_id, 0.0) / many, 2)

    def _switching_share(self) -> float:
        """What share of the working time went on loading models, as a percent."""
        counters = self.counters
        working = counters.switch_s + counters.served_s
        return round(100.0 * counters.switch_s / working, 1) if working else 0.0

    def _shapes_offered(self) -> list[dict]:
        """The kinds of request that can be sent here, and to which models.

        Two are in circulation. Nearly every client speaks the OpenAI one and
        every engine answers it. A client written against Anthropic's own
        library speaks the other, and only some engines do — so listing the
        base address alone would be half the answer, and the wrong half for
        anybody whose tool speaks the second.
        """
        answers: dict[str, list[str]] = {}
        engines_for: dict[str, set] = {}
        for entry in self.operations.configured():
            try:
                engine = self.operations.engines.get(entry["engine"])
                task = Task(entry.get("task", Task.TEXT_GENERATION.value))
                paths = engine.api_paths(task)
            except Exception:
                continue
            for path in paths:
                answers.setdefault(path, []).append(entry["id"])
                # The engines, as well as the models. A shape only some answer
                # is answered by an engine, not by a list of names that grows
                # every time an entry is added — "vLLM models" says it once and
                # stays true.
                engines_for.setdefault(path, set()).add(
                    getattr(engine, "display_name", entry["engine"]))
        return [{"path": path, "models": sorted(models),
                  "engines": sorted(engines_for[path])}
                for path, models in sorted(answers.items())]


def _runs(waiting: list[dict]) -> list[dict]:
    """The queue as the sequence of runs it will be served in.

    The queue is served in order and requests next to each other wanting the
    same model go in together, so it is not a list of requests — it is a list
    of turns, each one a model and how many. Grouping them the same way the
    scheduler does makes the page show the schedule rather than a number.
    """
    runs: list[dict] = []
    for item in waiting:
        shape = item["shape"]
        name = getattr(shape, "instance_id", str(shape))
        if runs and runs[-1]["instance_id"] == name:
            runs[-1]["requests"] += 1
            runs[-1]["longest_wait_s"] = max(runs[-1]["longest_wait_s"],
                                             item["waiting_s"])
            continue
        runs.append({"instance_id": name, "requests": 1,
                     "longest_wait_s": item["waiting_s"]})
    return runs


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
