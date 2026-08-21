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

from ai_lab.engines.base import ANTHROPIC_PATHS, OPENAI_PATHS
from ai_lab.gateway import (CardBusy, CouldNotLoad, Gateway, NotConfigured,
                            ShapeNotServed)
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


class FakeEngine:
    """An engine that answers a stated set of request shapes, and nothing else."""

    def __init__(self, shapes):
        self._shapes = shapes

    def api_paths(self):
        return self._shapes


class FakeRegistry:
    """The two engines as they really differ: only one speaks both shapes."""

    ENGINES = {
        "llamacpp": FakeEngine(OPENAI_PATHS),
        "vllm": FakeEngine(OPENAI_PATHS + ANTHROPIC_PATHS),
    }

    def get(self, engine_id):
        try:
            return self.ENGINES[engine_id]
        except KeyError:
            raise KeyError(f"Unknown engine: {engine_id}") from None


class FakeOperations:
    """Just enough of Operations to drive the gateway."""

    def __init__(self, entries, lag=0):
        self.engines = FakeRegistry()
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


def entry(identifier, name, model_id, port, running=False, engine="llamacpp"):
    return {"id": identifier, "name": name, "model_id": model_id, "port": port,
            "engine": engine, "running": running, "ready": running,
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

    def test_an_unknown_name_reads_as_a_sentence(self):
        # It is a KeyError so the web layer answers 404 on its own, and a
        # KeyError renders its argument with repr() — which would deliver this
        # whole sentence wrapped in quotes to somebody debugging an agent.
        with self.assertRaises(NotConfigured) as caught:
            self.gateway.resolve("nope")
        self.assertFalse(str(caught.exception).startswith(("'", '"')),
                         f"quoted: {str(caught.exception)!r}")

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

    def test_what_is_on_the_card_survives_an_unload_it_was_not_told_about(self):
        """Found on the real machine, not here.

        A person forced past a busy card from the page, which stops a model
        without going through the gateway — by design. The gateway went on
        naming that model as the one on the card, so the page showed a model
        that was no longer loaded, next to an accelerator reading empty.
        """
        operations = two_models()
        gateway = quick(operations)
        with gateway.acquire("coder"):
            pass
        self.assertEqual(gateway.stats()["current"], "coder")

        operations.unload("coder")              # straight past the gateway
        self.assertIsNone(gateway.stats()["current"],
                          "it reported a model that had been stopped behind its back")

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

    def test_the_card_reads_busy_while_a_model_is_still_loading(self):
        # Loading is when someone is most likely to look at this, and reporting
        # "not busy" then is the one answer that is certainly wrong.
        operations = two_models()
        operations.load_delay_s = 0.2
        gateway = quick(operations)
        seen = []

        def ask():
            with gateway.acquire("coder"):
                pass

        thread = threading.Thread(target=ask)
        thread.start()
        time.sleep(0.05)
        seen.append(gateway.stats()["busy"])
        thread.join(timeout=5)
        self.assertEqual(seen, [True])
        self.assertFalse(gateway.stats()["busy"])

    def test_the_card_is_reported_free_between_requests(self):
        gateway = quick(two_models(coder=True))
        with gateway.acquire("coder"):
            self.assertTrue(gateway.stats()["busy"])
        self.assertFalse(gateway.stats()["busy"])


if __name__ == "__main__":
    unittest.main()


class OneModelOnlyTests(unittest.TestCase):
    """One model on the card, with no exceptions.

    The rule is easy to state and easy to break in one particular way: a
    request that needs no switch used to leave the card exactly as it found it,
    strays and all.
    """

    def test_a_stray_is_unloaded_even_when_no_switch_is_needed(self):
        # Both are up — a manager restart found two units enabled, or somebody
        # pressed Load twice. The request asks for one of them, so nothing has
        # to be loaded; the other still has to go.
        operations = two_models(coder=True, reviewer=True)
        gateway = quick(operations)

        with gateway.acquire("coder") as lease:
            self.assertEqual(lease.port, 8080)

        self.assertEqual(operations.unloads, ["reviewer"])
        self.assertEqual(operations.loads, [],
                         "the model asked for was already up; nothing to load")
        running = [e["id"] for e in operations.instances() if e["running"]]
        self.assertEqual(running, ["coder"])

    def test_tidying_up_does_not_reload_the_model_that_was_already_there(self):
        # The cheap mistake is to treat "something else is running" as a switch
        # and put the wanted model through an unload and a load for nothing.
        operations = two_models(coder=True, reviewer=True)
        gateway = quick(operations)
        with gateway.acquire("coder"):
            pass
        self.assertNotIn("coder", operations.unloads)

    def test_a_tidy_up_is_recorded_as_what_it_was(self):
        operations = two_models(coder=True, reviewer=True)
        gateway = quick(operations)
        with gateway.acquire("coder"):
            pass
        entry = gateway.stats()["recent"][0]
        self.assertTrue(entry["tidied"])
        self.assertEqual(entry["unloaded"], ["reviewer"])
        self.assertEqual(gateway.stats()["switches"], 0,
                         "nothing was switched, so nothing should be counted")

    def test_a_clean_card_is_left_alone(self):
        operations = two_models(coder=True)
        gateway = quick(operations)
        with gateway.acquire("coder"):
            pass
        self.assertEqual(operations.unloads, [])
        self.assertEqual(gateway.stats()["recent"], [])


class BusyGuardTests(unittest.TestCase):
    """What the buttons on the page are told while a request is in progress."""

    def test_nothing_is_refused_while_the_card_is_free(self):
        gateway = quick(two_models(coder=True))
        self.assertIsNone(gateway.busy())
        gateway.guard("unload", "coder")        # does not raise

    def test_stopping_a_model_that_is_answering_is_refused(self):
        gateway = quick(two_models(coder=True))
        with gateway.acquire("coder"):
            with self.assertRaises(CardBusy) as caught:
                gateway.guard("unload", "coder")
        self.assertIn("coder", str(caught.exception))
        self.assertTrue(caught.exception.detail["busy"]["answering"])

    def test_the_refusal_names_who_holds_the_card(self):
        # The page cannot offer a useful choice without knowing which model is
        # working — the one being stopped may not be the one that is busy.
        gateway = quick(two_models(coder=True))
        with gateway.acquire("coder"):
            with self.assertRaises(CardBusy) as caught:
                gateway.guard("load", "reviewer")
        self.assertEqual(caught.exception.detail["busy"]["instance_id"], "coder")

    def test_the_card_is_free_again_once_the_answer_is_finished(self):
        gateway = quick(two_models(coder=True))
        with gateway.acquire("coder"):
            pass
        self.assertIsNone(gateway.busy())
        gateway.guard("unload", "coder")        # does not raise

    def test_a_model_still_loading_counts_as_busy(self):
        # The card is taken and no answer is being written yet, but a request
        # is already waiting on that load. Interrupting it is still an
        # interruption.
        operations = two_models()
        operations.load_delay_s = 0.2
        gateway = quick(operations)
        seen = []

        def request():
            with gateway.acquire("coder"):
                pass

        thread = threading.Thread(target=request)
        thread.start()
        time.sleep(0.05)
        try:
            gateway.guard("unload", "coder")
        except CardBusy as error:
            seen.append(error.detail["busy"])
        thread.join(timeout=5)

        self.assertEqual(len(seen), 1, "a load in progress should be refused")
        self.assertFalse(seen[0]["answering"],
                         "it is loading, not answering, and should say so")

    def test_a_failed_request_does_not_leave_the_card_looking_busy(self):
        operations = two_models()

        def refuse(instance_id):
            return Operation(instance_id=instance_id, kind="load", ok=False,
                             error="no")
        operations.load = refuse

        gateway = quick(operations)
        with self.assertRaises(CouldNotLoad):
            gateway.acquire("coder")
        self.assertIsNone(gateway.busy())
        gateway.guard("unload", "coder")        # does not raise


class OverheadTests(unittest.TestCase):
    """How many times one request asks what the instances are doing.

    Measured on the container: that question costs about 125 ms and 28
    processes, because it asks the supervisor about every configured instance
    and probes each one that is up. A request to an engine that answers in
    17 ms was taking 500 ms to get there, all of it spent asking the same
    question over and over.

    The count is asserted rather than the milliseconds: a timing test on a
    laptop measures the laptop.
    """

    def counting(self, **running):
        operations = two_models(**running)
        operations.reads = 0
        original = operations.instances

        def counted():
            operations.reads += 1
            return original()
        operations.instances = counted
        return operations

    def test_a_request_to_a_loaded_model_asks_twice(self):
        # Once to resolve the name before queueing, once after the card is
        # taken — the request in front may have changed what is loaded while
        # this one waited, so the second read cannot be skipped.
        operations = self.counting(coder=True)
        gateway = quick(operations)
        with gateway.acquire("coder"):
            pass
        self.assertEqual(operations.reads, 2)

    def test_the_engine_name_costs_no_extra_read(self):
        # It is pure configuration, and the lease carries it. Asking for it
        # afterwards meant asking the expensive question again for an answer
        # that was already in hand.
        operations = self.counting(coder=True)
        gateway = quick(operations)
        with gateway.acquire("coder") as lease:
            self.assertEqual(lease.model_name, "Qwen3.6-35B")
        self.assertEqual(operations.reads, 2)

    def test_tidying_up_costs_no_extra_read(self):
        operations = self.counting(coder=True, reviewer=True)
        gateway = quick(operations)
        with gateway.acquire("coder"):
            pass
        self.assertEqual(operations.reads, 2)
        self.assertEqual(operations.unloads, ["reviewer"])


def mixed_engines(**running):
    """One entry on each engine, which is the whole point of these tests."""
    return FakeOperations([
        entry("coder", "Coding", "gguf/qwen/Qwen3.6-35B", 8080,
              running.get("coder", False), engine="llamacpp"),
        entry("fast", "Fast", "nvfp4/qwen3-coder-30b", 8082,
              running.get("fast", False), engine="vllm"),
    ])


class RequestShapeTests(unittest.TestCase):
    """Two ways of writing the same request, and not every engine reads both.

    Nearly everything speaks the OpenAI shape. A client written against
    Anthropic's own library sends `/v1/messages` instead. vLLM answers that
    one; llama.cpp does not.

    The engine says which it answers, so nothing above has to know one engine
    from another — and adding a third shape, or an engine that speaks it, is a
    line in that engine's file.
    """

    def test_the_usual_shape_works_on_either_engine(self):
        gateway = quick(mixed_engines(coder=True))
        with gateway.acquire("coder", shape="/v1/chat/completions") as lease:
            self.assertEqual(lease.port, 8080)

    def test_the_other_shape_works_where_the_engine_answers_it(self):
        gateway = quick(mixed_engines(fast=True))
        with gateway.acquire("fast", shape="/v1/messages") as lease:
            self.assertEqual(lease.port, 8082)

    def test_the_other_shape_is_refused_where_it_is_not_answered(self):
        gateway = quick(mixed_engines(coder=True))
        with self.assertRaises(ShapeNotServed) as caught:
            gateway.acquire("coder", shape="/v1/messages")
        self.assertIn("llamacpp", str(caught.exception))

    def test_the_refusal_names_the_models_that_would_have_worked(self):
        # A client does not know which of its models is on which engine, and a
        # refusal that only says no leaves it with nowhere to go.
        gateway = quick(mixed_engines(coder=True))
        with self.assertRaises(ShapeNotServed) as caught:
            gateway.acquire("coder", shape="/v1/messages")
        self.assertIn("fast", str(caught.exception))

    def test_it_says_so_plainly_when_nothing_answers_the_shape(self):
        operations = FakeOperations([
            entry("only", "Only", "gguf/a/a", 8080, True, engine="llamacpp")])
        gateway = quick(operations)
        with self.assertRaises(ShapeNotServed) as caught:
            gateway.acquire("only", shape="/v1/messages")
        self.assertIn("No configured model answers it", str(caught.exception))

    def test_nothing_is_loaded_for_a_shape_that_will_be_refused(self):
        # The refusal happens before the card is touched. Loading a model for
        # forty seconds and then saying no would be the worst of both.
        operations = mixed_engines()
        gateway = quick(operations)
        with self.assertRaises(ShapeNotServed):
            gateway.acquire("coder", shape="/v1/messages")
        self.assertEqual(operations.loads, [])
        self.assertEqual(operations.unloads, [])

    def test_a_refused_shape_does_not_leave_the_card_taken(self):
        operations = mixed_engines(coder=True)
        gateway = quick(operations)
        with self.assertRaises(ShapeNotServed):
            gateway.acquire("coder", shape="/v1/messages")
        self.assertIsNone(gateway.busy())
        with gateway.acquire("coder", shape="/v1/chat/completions"):
            pass

    def test_no_shape_given_means_no_check(self):
        # The card can be taken for reasons that are not a forwarded request.
        gateway = quick(mixed_engines(coder=True))
        with gateway.acquire("coder"):
            pass

    def test_the_listing_says_which_shapes_each_entry_answers(self):
        gateway = quick(mixed_engines())
        rows = {row["id"]: row for row in gateway.catalogue()}
        self.assertIn("/v1/messages", rows["fast"]["shapes"])
        self.assertNotIn("/v1/messages", rows["coder"]["shapes"])
        self.assertIn("/v1/chat/completions", rows["coder"]["shapes"])
