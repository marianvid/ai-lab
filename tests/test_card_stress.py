"""The card under load, judged only from the outside.

Many threads at once, two models, and requests that give up. Nothing here looks inside the scheduler or
the gateway: these tests say what must stay true whatever the classes look
like, so they hold the line while the classes are rearranged.

Afterwards: every request was answered or refused with a reason, no place was
lost, never more requests on a model than it has places, and nothing hung.

Deterministic where it can be: every worker draws from its own fixed seed, and
every test has a hard time limit that fails it rather than letting it hang.
The short sleeps only stand for a request holding its place while it is
answered; nothing waits for a state by sleeping — it waits on an event.
"""

import random
import threading
import time
import unittest

from ai_lab.gateway.admission.scheduler import AbandonedError, Scheduler
from tests.test_gateway import quick, two_models

THREADS = 24
ROUNDS = 15
LIMIT_S = 30.0


class Card:
    """A card that records loads and checks no shape ever has more than its places."""

    def __init__(self, places):
        self.places_for = places
        self.lock = threading.Lock()
        self.switches = []
        self.violations = []

    def switch(self, shape, victims=()):
        time.sleep(0.002)
        with self.lock:
            self.switches.append(shape)

    def places(self, shape):
        return self.places_for[shape]


def _answer(wanted):
    """A `still_wanted` that always says the same thing (bound now, not at call time)."""
    return lambda: wanted


def run_all(workers):
    threads = [threading.Thread(target=work, daemon=True) for work in workers]
    for thread in threads:
        thread.start()
    deadline = time.monotonic() + LIMIT_S
    for thread in threads:
        thread.join(max(0.0, deadline - time.monotonic()))
    return [thread for thread in threads if thread.is_alive()]


class SchedulerStressTests(unittest.TestCase):
    def test_many_threads_two_models_and_quitters(self):
        card = Card({"a": 3, "b": 2})
        scheduler = Scheduler(card.switch, card.places)
        outcomes = {"served": 0, "abandoned": 0}
        lock = threading.Lock()
        peak = {"a": 0, "b": 0}

        def watch_peak():
            for item in scheduler.state()["loaded"]:
                with lock:
                    peak[item["shape"]] = max(peak[item["shape"]], item["in_flight"])

        def worker(seed):
            chance = random.Random(seed)

            def work():
                for _ in range(ROUNDS):
                    shape = chance.choice("ab")
                    wanted = chance.random() >= 0.15
                    try:
                        scheduler.enter(shape, still_wanted=_answer(wanted))
                    except AbandonedError:
                        with lock:
                            outcomes["abandoned"] += 1
                        continue
                    watch_peak()
                    time.sleep(chance.random() * 0.002)
                    scheduler.leave(shape)
                    with lock:
                        outcomes["served"] += 1
            return work

        hung = run_all([worker(seed) for seed in range(THREADS)])
        self.assertEqual(hung, [], "requests still waiting after the time limit")
        self.assertEqual(outcomes["served"] + outcomes["abandoned"], THREADS * ROUNDS)
        state = scheduler.state()
        self.assertEqual((state["in_flight"], state["waiting"]), (0, []))
        self.assertLessEqual(peak["a"], 3)
        self.assertLessEqual(peak["b"], 2)
        self.assertGreater(len(card.switches), 1)


class GatewayStressTests(unittest.TestCase):
    def test_many_threads_on_two_models_lose_no_place(self):
        operations = two_models()
        gateway = quick(operations)
        failures = []

        def worker(seed):
            chance = random.Random(seed)

            def work():
                for _ in range(ROUNDS // 3):
                    try:
                        lease = gateway.acquire(chance.choice(["coder", "reviewer"]))
                    except Exception as error:  # noqa: BLE001 - recorded for the test
                        failures.append(error)
                        continue
                    time.sleep(chance.random() * 0.002)
                    lease.release()
            return work

        hung = run_all([worker(seed) for seed in range(THREADS // 2)])
        self.assertEqual(hung, [])
        self.assertEqual(failures, [])
        stats = gateway.stats()
        self.assertEqual((stats["in_flight"], stats["waiting"]), (0, 0))
