"""The gateway, from an agent's side.

Each test names something an agent workflow would hit. The card is faked: what
matters is which entry gets loaded, what gets unloaded first, and that a request
in progress cannot have its model pulled out from under it.

The design being tested is sequential. One model on the card, one request at a
time, and a switch empties the card completely before loading.
"""

import threading
import time
import unittest

from ai_lab.gateway import CouldNotLoad, Gateway, NotConfigured
from ai_lab.runtime import Operation
from ai_lab.types import AcceleratorSnapshot


class FakeHost:
    """A card whose memory reading follows what is loaded.

    Real hardware returns the memory a moment after a process exits. `lag`
    reproduces that: the first few readings after an unload still show the model
    on the card, so the gateway has something real to wait for.
    """

    def __init__(self, operations, lag=0):
        self.operations = operations
        self.lag = lag
        self.readings = 0

    def accelerator(self):
        self.readings += 1
        running = any(e["running"] for e in self.operations.instances())
        stale = self.lag > 0
        if stale:
            self.lag -= 1
        used = 20000.0 if (running or stale) else 2.0
        return AcceleratorSnapshot(available=True, name="Fake", kind="cuda",
                                   memory_kind="dedicated",
                                   memory_used_mb=used, memory_total_mb=32000.0)


class FakeOperations:
    """Just enough of Operations to drive the gateway."""

    def __init__(self, entries, lag=0):
        self._entries = {e["id"]: dict(e) for e in entries}
        self.loads = []
        self.unloads = []
        self.load_delay_s = 0.0
        self.host = FakeHost(self, lag=lag)

    def instances(self):
        return [dict(entry) for entry in self._entries.values()]

    def load(self, instance_id):
        self.loads.append(instance_id)
        if self.load_delay_s:
            time.sleep(self.load_delay_s)
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


def two_models(lag=0, **running):
    return FakeOperations([
        entry("coder", "Coding", "gguf/qwen/Qwen3.6-35B", 8080,
              running.get("coder", False)),
        entry("reviewer", "Review", "nvfp4/qwopus-27b", 8083,
              running.get("reviewer", False)),
    ], lag=lag)


def quick(operations):
    """A gateway that does not sit through real waits in a test."""
    return Gateway(operations, quiet_timeout_s=1.0, poll_s=0.001)


class NamingTests(unittest.TestCase):
    def setUp(self):
        self.gateway = quick(two_models())

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
        gateway = quick(operations)
        with gateway.acquire("coder") as lease:
            self.assertEqual(lease.port, 8080)
        self.assertEqual(operations.loads, [])
        self.assertEqual(operations.unloads, [])

    def test_asking_for_a_stopped_model_loads_it(self):
        operations = two_models()
        gateway = quick(operations)
        with gateway.acquire("coder") as lease:
            self.assertEqual(lease.port, 8080)
        self.assertEqual(operations.loads, ["coder"])

    def test_a_different_model_empties_the_card_first(self):
        operations = two_models(coder=True)
        gateway = quick(operations)
        with gateway.acquire("reviewer"):
            pass
        self.assertEqual(operations.unloads, ["coder"])
        self.assertEqual(operations.loads, ["reviewer"])

    def test_only_the_new_model_is_left_running(self):
        # Not "one fewer than before" — one, full stop.
        operations = two_models(coder=True, reviewer=True)
        gateway = quick(operations)
        operations._entries["third"] = entry("third", "Third", "gguf/x/third", 8085)
        with gateway.acquire("third"):
            pass
        running = {i["id"] for i in operations.instances() if i["running"]}
        self.assertEqual(running, {"third"})

    def test_everything_running_is_unloaded_not_only_the_last_one_asked_for(self):
        # A manager restart can leave more than one engine up. Loading on top of
        # whatever is left is how a model that is known to fit stops fitting.
        operations = two_models(coder=True, reviewer=True)
        gateway = quick(operations)
        operations._entries["third"] = entry("third", "Third", "gguf/x/third", 8085)
        with gateway.acquire("third"):
            pass
        self.assertEqual(sorted(operations.unloads), ["coder", "reviewer"])

    def test_the_load_waits_until_the_card_has_actually_gone_quiet(self):
        # The driver hands memory back after the process exits, not with it.
        operations = two_models(lag=3, coder=True)
        gateway = quick(operations)
        with gateway.acquire("reviewer"):
            pass
        self.assertEqual(operations.loads, ["reviewer"])
        # It polled rather than loading straight into memory still held.
        self.assertGreater(operations.host.readings, 2)

    def test_a_card_that_never_goes_quiet_fails_with_a_readable_reason(self):
        operations = two_models(lag=10_000, coder=True)
        gateway = quick(operations)
        with self.assertRaises(CouldNotLoad) as caught:
            gateway.acquire("reviewer")
        self.assertIn("still holds", str(caught.exception))
        self.assertEqual(operations.loads, [])

    def test_unified_memory_is_not_waited_on(self):
        # On an M3 Max there is no separate pool to come back.
        operations = two_models(coder=True)
        operations.host.accelerator = lambda: AcceleratorSnapshot(
            available=True, name="M3 Max", kind="metal", memory_kind="unified",
            memory_used_mb=48000.0, memory_total_mb=96000.0)
        gateway = quick(operations)
        with gateway.acquire("reviewer"):
            pass
        self.assertEqual(operations.loads, ["reviewer"])

    def test_a_model_that_will_not_start_raises_rather_than_hanging(self):
        operations = two_models()

        def refuse(instance_id):
            operations.loads.append(instance_id)
            return Operation(instance_id=instance_id, kind="load", ok=False,
                             error="engine died during startup")
        operations.load = refuse
        gateway = quick(operations)
        with self.assertRaises(CouldNotLoad) as caught:
            gateway.acquire("coder")
        self.assertIn("engine died", str(caught.exception))

    def test_a_failed_load_does_not_leave_the_gateway_blocked(self):
        # A switch that raises must release the card, or every later request
        # waits forever on a switch that already gave up.
        operations = two_models()
        original = operations.load
        operations.load = lambda i: Operation(instance_id=i, kind="load",
                                              ok=False, error="boom")
        gateway = quick(operations)
        with self.assertRaises(CouldNotLoad):
            gateway.acquire("coder")
        operations.load = original
        with gateway.acquire("coder") as lease:
            self.assertEqual(lease.port, 8080)


class SequenceTests(unittest.TestCase):
    def test_a_switch_waits_for_the_request_that_is_still_running(self):
        # An agent streaming a long answer must not have its model unloaded
        # underneath it.
        operations = two_models(coder=True)
        gateway = quick(operations)
        order = []

        lease = gateway.acquire("coder")

        def switch():
            with gateway.acquire("reviewer"):
                order.append("switched")

        thread = threading.Thread(target=switch)
        thread.start()
        time.sleep(0.05)
        order.append("first request still going")
        lease.__exit__()
        thread.join(timeout=5)

        self.assertEqual(order, ["first request still going", "switched"])

    def test_two_requests_for_the_same_model_still_take_turns(self):
        # This is the deliberate consequence of the design: sequential means
        # sequential, even when no switch is needed.
        operations = two_models(coder=True)
        gateway = quick(operations)
        order = []
        first = gateway.acquire("coder")

        def second():
            with gateway.acquire("coder"):
                order.append("second")

        thread = threading.Thread(target=second)
        thread.start()
        time.sleep(0.05)
        order.append("first")
        first.__exit__()
        thread.join(timeout=5)
        self.assertEqual(order, ["first", "second"])

    def test_several_requests_arriving_together_load_the_model_once(self):
        operations = two_models()
        operations.load_delay_s = 0.02
        gateway = quick(operations)

        def ask():
            with gateway.acquire("coder"):
                time.sleep(0.01)

        threads = [threading.Thread(target=ask) for _ in range(5)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)
        self.assertEqual(operations.loads, ["coder"])
        self.assertEqual(operations.unloads, [])


class ReportingTests(unittest.TestCase):
    def test_switches_are_counted_so_thrashing_can_be_seen(self):
        operations = two_models(coder=True)
        gateway = quick(operations)
        for name in ("reviewer", "coder", "reviewer"):
            with gateway.acquire(name):
                pass
        stats = gateway.stats()
        self.assertEqual(stats["requests"], 3)
        self.assertEqual(stats["switches"], 3)

    def test_a_run_with_no_switching_reports_none(self):
        gateway = quick(two_models(coder=True))
        for _ in range(3):
            with gateway.acquire("coder"):
                pass
        self.assertEqual(gateway.stats()["switches"], 0)

    def test_the_current_model_is_reported(self):
        gateway = quick(two_models(coder=True))
        with gateway.acquire("reviewer"):
            pass
        self.assertEqual(gateway.stats()["current"], "reviewer")

    def test_recent_history_says_what_was_unloaded_for_what(self):
        gateway = quick(two_models(coder=True))
        with gateway.acquire("reviewer"):
            pass
        recent = gateway.stats()["recent"]
        self.assertEqual(recent[0]["loaded"], "reviewer")
        self.assertEqual(recent[0]["unloaded"], ["coder"])

    def test_handing_the_card_back_twice_does_not_free_the_next_request(self):
        # The forwarding code releases when the answer ends, and again if the
        # connection broke on the way out. The second one must do nothing.
        gateway = quick(two_models(coder=True))
        lease = gateway.acquire("coder")
        gateway.release()
        gateway.release()
        second = gateway.acquire("coder")
        self.assertTrue(gateway.stats()["busy"])
        second.__exit__()
        self.assertFalse(gateway.stats()["busy"])

    def test_the_card_is_reported_free_between_requests(self):
        gateway = quick(two_models(coder=True))
        with gateway.acquire("coder"):
            self.assertTrue(gateway.stats()["busy"])
        self.assertFalse(gateway.stats()["busy"])


if __name__ == "__main__":
    unittest.main()
