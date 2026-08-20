"""The gateway, from an agent's side.

Each test names something an agent workflow would hit and checks it behaves.
The card is faked: what matters here is which entry gets loaded, what gets
evicted, and what happens when two requests want different models at once.
"""

import threading
import time
import unittest

from ai_lab.gateway import CouldNotLoad, Gateway, NotConfigured
from ai_lab.runtime import Operation


class FakeOperations:
    """Just enough of Operations to drive the gateway.

    `capacity` is how many entries can be loaded at once. Asking for one more
    fails with the runtime's own wording, which is what the gateway matches on
    to decide that it must evict something.
    """

    def __init__(self, entries, capacity=1):
        self._entries = {e["id"]: dict(e) for e in entries}
        self.capacity = capacity
        self.loads = []
        self.unloads = []
        self.load_delay_s = 0.0

    def instances(self):
        return [dict(entry) for entry in self._entries.values()]

    def load(self, instance_id):
        self.loads.append(instance_id)
        if self.load_delay_s:
            time.sleep(self.load_delay_s)
        running = [e for e in self._entries.values()
                   if e["running"] and e["id"] != instance_id]
        if len(running) >= self.capacity:
            return Operation(instance_id=instance_id, kind="load", ok=False,
                             error="qwen needs about 20.0 GB but only 2.0 GB "
                                   "is free on the card. Unload another model.")
        self._entries[instance_id].update(running=True, ready=True)
        return Operation(instance_id=instance_id, kind="load", ok=True, total_ms=1200)

    def unload(self, instance_id):
        self.unloads.append(instance_id)
        self._entries[instance_id].update(running=False, ready=False)
        return Operation(instance_id=instance_id, kind="unload", ok=True)


def entry(identifier, name, model_id, port, running=False):
    return {"id": identifier, "name": name, "model_id": model_id, "port": port,
            "engine": "llamacpp", "running": running, "ready": running,
            "params": {}, "enabled": True, "pid": None, "web_ui": False,
            "last_operation": None}


def two_models(capacity=1, **running):
    return FakeOperations([
        entry("coder", "Coding", "gguf/qwen/Qwen3.6-35B", 8080,
              running.get("coder", False)),
        entry("reviewer", "Review", "nvfp4/qwopus-27b", 8083,
              running.get("reviewer", False)),
    ], capacity=capacity)


class NamingTests(unittest.TestCase):
    def setUp(self):
        self.gateway = Gateway(two_models())

    def test_a_model_answers_to_its_entry_id(self):
        self.assertEqual(self.gateway.resolve("coder")["id"], "coder")

    def test_a_model_answers_to_the_label_a_person_gave_it(self):
        self.assertEqual(self.gateway.resolve("Coding")["id"], "coder")

    def test_a_model_answers_to_the_name_its_engine_reports(self):
        # A client that read /v1/models on the engine itself has this name and
        # nothing else, so refusing it would be refusing the obvious.
        self.assertEqual(self.gateway.resolve("Qwen3.6-35B")["id"], "coder")

    def test_names_are_matched_regardless_of_case(self):
        self.assertEqual(self.gateway.resolve("cODiNg")["id"], "coder")

    def test_an_unknown_name_says_what_is_known(self):
        with self.assertRaises(NotConfigured) as caught:
            self.gateway.resolve("gpt-4")
        message = str(caught.exception)
        self.assertIn("gpt-4", message)
        self.assertIn("Coding", message)

    def test_the_catalogue_lists_models_that_are_not_loaded(self):
        # The whole point is that a client may ask for one of these.
        rows = self.gateway.catalogue()
        self.assertEqual(len(rows), 2)
        self.assertFalse(any(row["loaded"] for row in rows))


class LoadingTests(unittest.TestCase):
    def test_asking_for_a_loaded_model_does_not_reload_it(self):
        operations = two_models(coder=True)
        gateway = Gateway(operations)
        with gateway.acquire("coder") as lease:
            self.assertEqual(lease.port, 8080)
        self.assertEqual(operations.loads, [])

    def test_asking_for_a_stopped_model_loads_it(self):
        operations = two_models()
        gateway = Gateway(operations)
        with gateway.acquire("coder") as lease:
            self.assertEqual(lease.port, 8080)
        self.assertEqual(operations.loads, ["coder"])

    def test_a_second_model_evicts_the_first_when_only_one_fits(self):
        operations = two_models(coder=True)
        gateway = Gateway(operations)
        with gateway.acquire("reviewer"):
            pass
        self.assertEqual(operations.unloads, ["coder"])
        self.assertIn("reviewer", operations.loads)

    def test_both_stay_loaded_when_the_card_has_room(self):
        # Two small models fit together, so a workflow alternating between them
        # should pay for loading once each and never swap again.
        operations = two_models(capacity=2)
        gateway = Gateway(operations)
        for name in ("coder", "reviewer", "coder", "reviewer"):
            with gateway.acquire(name):
                pass
        self.assertEqual(operations.unloads, [])
        self.assertEqual(sorted(operations.loads), ["coder", "reviewer"])

    def test_the_least_recently_used_model_is_the_one_evicted(self):
        operations = two_models(capacity=2)
        gateway = Gateway(operations)
        with gateway.acquire("coder"):
            pass
        time.sleep(0.01)
        with gateway.acquire("reviewer"):
            pass
        operations.capacity = 1        # the card is now full
        operations._entries["third"] = entry("third", "Third", "gguf/x/third", 8085)
        with gateway.acquire("third"):
            pass
        # Both had to go to make room for one, and the order is what is being
        # tested: coder was used first, so coder goes first.
        self.assertEqual(operations.unloads[0], "coder")

    def test_a_model_that_will_not_start_raises_rather_than_hanging(self):
        operations = two_models()

        def refuse(instance_id):
            operations.loads.append(instance_id)
            return Operation(instance_id=instance_id, kind="load", ok=False,
                             error="engine died during startup")
        operations.load = refuse
        gateway = Gateway(operations)
        with self.assertRaises(CouldNotLoad):
            gateway.acquire("coder")

    def test_a_failed_load_does_not_leave_the_gateway_blocked(self):
        # A swap that raises must clear its own flag, or every later request
        # waits for a swap that already gave up.
        operations = two_models()
        original = operations.load
        operations.load = lambda i: Operation(instance_id=i, kind="load",
                                              ok=False, error="boom")
        gateway = Gateway(operations)
        with self.assertRaises(CouldNotLoad):
            gateway.acquire("coder")
        operations.load = original
        with gateway.acquire("coder") as lease:
            self.assertEqual(lease.port, 8080)


class ConcurrencyTests(unittest.TestCase):
    def test_a_swap_waits_for_a_request_that_is_still_running(self):
        # An agent streaming a long answer must not have its model unloaded
        # underneath it.
        operations = two_models(coder=True)
        gateway = Gateway(operations)
        order = []

        lease = gateway.acquire("coder")

        def switch():
            with gateway.acquire("reviewer"):
                order.append("swapped")

        thread = threading.Thread(target=switch)
        thread.start()
        time.sleep(0.05)
        order.append("first request still going")
        lease.__exit__()
        thread.join(timeout=5)

        self.assertEqual(order, ["first request still going", "swapped"])

    def test_two_requests_for_the_same_loaded_model_run_at_once(self):
        # Serialising these would throw away everything vLLM's batching buys.
        operations = two_models(coder=True)
        gateway = Gateway(operations)
        first = gateway.acquire("coder")
        second = gateway.acquire("coder")
        self.assertEqual(gateway.stats()["in_flight"], 2)
        first.__exit__()
        second.__exit__()
        self.assertEqual(gateway.stats()["in_flight"], 0)

    def test_only_one_load_happens_when_several_requests_arrive_together(self):
        operations = two_models()
        operations.load_delay_s = 0.05
        gateway = Gateway(operations)

        def ask():
            with gateway.acquire("coder"):
                time.sleep(0.01)

        threads = [threading.Thread(target=ask) for _ in range(5)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)
        self.assertEqual(operations.loads, ["coder"])


class ReportingTests(unittest.TestCase):
    def test_swaps_are_counted_so_thrashing_can_be_seen(self):
        operations = two_models(coder=True)
        gateway = Gateway(operations)
        for name in ("reviewer", "coder", "reviewer"):
            with gateway.acquire(name):
                pass
        stats = gateway.stats()
        self.assertEqual(stats["requests"], 3)
        self.assertEqual(stats["swaps"], 3)
        self.assertEqual(stats["evictions"], 3)

    def test_a_run_with_no_swapping_reports_none(self):
        gateway = Gateway(two_models(coder=True))
        for _ in range(3):
            with gateway.acquire("coder"):
                pass
        self.assertEqual(gateway.stats()["swaps"], 0)

    def test_recent_history_says_what_was_evicted_for_what(self):
        gateway = Gateway(two_models(coder=True))
        with gateway.acquire("reviewer"):
            pass
        recent = gateway.stats()["recent"]
        self.assertEqual(recent[0]["loaded"], "reviewer")
        self.assertEqual(recent[0]["evicted"], ["coder"])


if __name__ == "__main__":
    unittest.main()
