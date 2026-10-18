"""The card under load, judged only from the outside.

Many threads at once, two models, requests that give up, and card reservations
granted and released during the run. Nothing here looks inside the scheduler or
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

from ai_lab.api.routes.gateway import _Admission, _take_place
from ai_lab.gateway.scheduler import AbandonedError, Scheduler
from ai_lab.reservations.book import ReservationBook
from ai_lab.reservations.errors import CardReservedError
from ai_lab.reservations.settings import ReservationPolicy
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


class ReservationStressTests(unittest.TestCase):
    """A holder keeps its model on the card while others keep asking for another."""

    def test_no_other_model_is_loaded_while_the_card_is_held(self):
        operations = two_models()
        gateway = quick(operations)
        book = ReservationBook(ReservationPolicy(unreserved_wait_s=5))
        held = threading.Event()
        released = threading.Event()
        window = {}
        refused = []

        def holder():
            made = book.request({"team": "ingest", "priority": 4})
            admission = _Admission(book, made["id"], None)
            for _ in range(10):
                admission.at_the_door()
                lease = _take_place(gateway, "/v1/chat/completions",
                                    {"model": "coder"}, admission)
                if not held.is_set():
                    window["from"] = len(operations.loads)
                    held.set()
                time.sleep(0.003)
                lease.release()
            window["to"] = len(operations.loads)
            book.release(made["id"])
            released.set()

        def other(seed):
            chance = random.Random(seed)

            def work():
                held.wait(5)
                for _ in range(6):
                    admission = _Admission(book, None, None)
                    try:
                        admission.at_the_door()
                        lease = _take_place(gateway, "/v1/chat/completions",
                                            {"model": "reviewer"}, admission)
                    except CardReservedError as error:
                        refused.append(error)
                        continue
                    time.sleep(chance.random() * 0.002)
                    lease.release()
            return work

        hung = run_all([holder] + [other(seed) for seed in range(6)])
        self.assertEqual(hung, [])
        self.assertTrue(released.is_set())
        loads_while_held = operations.loads[window["from"]:window["to"]]
        self.assertNotIn("reviewer", loads_while_held)
        self.assertIn("reviewer", operations.loads[window["to"]:])
        self.assertEqual(gateway.stats()["in_flight"], 0)
