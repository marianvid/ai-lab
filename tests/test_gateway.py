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

from ai_lab.engines.base import (ANTHROPIC_PATHS, OPENAI_PATHS, ParamSpec,
                                 validate)
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

    def system_memory(self):
        """A machine with a separate pool, so the page has something to draw."""
        return 8000.0, 64000.0

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


# The settings a fake engine takes. Few, but real ParamSpecs checked by the
# real `validate`, so a test about refusing a bad setting is testing the rule
# the interface uses rather than a stand-in for it.
FAKE_PARAMS = (
    ParamSpec("context_size", "Context size", "int", 32768,
              minimum=512, maximum=1048576, group="memory", help="How much."),
    ParamSpec("parallel", "Slots", "int", 1, minimum=1, maximum=64,
              group="memory", help="How many."),
)


class FakeEngine:
    """An engine that answers a stated set of request shapes, and nothing else."""

    def __init__(self, shapes):
        self._shapes = shapes

    def api_paths(self):
        return self._shapes

    def params(self):
        return FAKE_PARAMS

    def concurrency(self, params):
        """The real engines each spell this differently; the fake uses one name.

        What matters here is that the number comes from the engine rather than
        from the gateway guessing at a setting name.
        """
        return max(1, int(validate(FAKE_PARAMS, params)["parallel"]))


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
        self.cheap_reads = 0
        self.load_settings = []
        self.unloads = []
        self.load_delay_s = 0.0
        self.host = FakeHost(self, lag=lag)

    def instances(self):
        return [dict(entry) for entry in self._entries.values()]

    def load(self, instance_id, settings=None):
        self.loads.append(instance_id)
        self.load_settings.append(settings)
        if self.load_delay_s:
            time.sleep(self.load_delay_s)
        entry = self._entries[instance_id]
        # What it is running with, when that is not what it is configured with
        # — the same thing the real runtime reports.
        active = {} if settings is None else {
            key: value for key, value in settings.items()
            if entry["params"].get(key) != value}
        entry.update(running=True, ready=True, active_params=active)
        return Operation(instance_id=instance_id, kind="load", ok=True, total_ms=1200)

    def configured(self):
        """Every entry, configuration only — no supervisor, no probe.

        Counted apart from `instances`, because the point of it is that it is
        the cheap question and the front door asks it on every request.
        """
        self.cheap_reads += 1
        return [{key: entry[key] for key in
                 ("id", "name", "engine", "model_id", "port", "params")}
                for entry in self._entries.values()]

    def instance(self, instance_id):
        """One entry, configuration only — no supervisor, no probe.

        The real one reads the configuration file; this reads the same
        dictionary the fake keeps. Counted separately from `instances`, because
        the whole point of it is that it is the cheap question.
        """
        self.cheap_reads += 1
        entry = self._entries[instance_id]
        return {key: entry[key] for key in
                ("id", "name", "engine", "model_id", "port", "params")}

    def effective_params(self, instance_id, settings):
        entry = self._entries[instance_id]
        engine = self.engines.get(entry["engine"])
        return validate(engine.params(), {**entry["params"], **settings})

    def unload(self, instance_id):
        self.unloads.append(instance_id)
        self._entries[instance_id].update(running=False, ready=False,
                                          active_params={})
        return Operation(instance_id=instance_id, kind="unload", ok=True)


def entry(identifier, name, model_id, port, running=False, engine="llamacpp"):
    return {"id": identifier, "name": name, "model_id": model_id, "port": port,
            "engine": engine, "running": running, "ready": running,
            "params": {"context_size": 32768, "parallel": 1},
            "active_params": {},
            "enabled": True, "pid": None, "web_ui": False,
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

        def refuse(instance_id, settings=None):
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
        operations.load = lambda i, s=None: Operation(instance_id=i, kind="load",
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
        operations = two_models()
        gateway = quick(operations)
        for wanted in ("coder", "reviewer", "coder"):
            with gateway.acquire(wanted):
                pass
        stats = gateway.stats()
        self.assertEqual(stats["switches"], 3)
        self.assertGreaterEqual(stats["requests_per_minute"], 3)

    def test_a_lifetime_total_of_requests_is_not_reported(self):
        # It grows while you watch it and means the same at 40 as at 40,000.
        # A rate stays comparable to itself.
        gateway = quick(two_models(coder=True))
        with gateway.acquire("coder"):
            pass
        self.assertNotIn("requests", gateway.stats())

    def test_the_share_of_working_time_spent_loading(self):
        # Against the wall clock, a machine idle overnight reports a
        # flattering number for a workflow that spends its life swapping. This
        # is of the time it was working — answering or loading.
        gateway = quick(two_models())
        with gateway.acquire("coder"):
            pass
        share = gateway.stats()["switching_share"]
        self.assertGreater(share, 0.0)
        self.assertLessEqual(share, 100.0)

    def test_nothing_working_is_reported_as_no_share_rather_than_a_crash(self):
        self.assertEqual(quick(two_models()).stats()["switching_share"], 0.0)

    def test_time_to_the_first_token_is_averaged(self):
        gateway = quick(two_models(coder=True))
        gateway.first_token(0.4)
        gateway.first_token(0.6)
        self.assertEqual(gateway.stats()["average_first_token_s"], 0.5)

    def test_no_streamed_request_yet_reports_nothing_rather_than_dividing(self):
        self.assertEqual(quick(two_models()).stats()["average_first_token_s"], 0.0)

    def test_a_run_with_no_switching_reports_none(self):
        gateway = quick(two_models(coder=True))
        for _ in range(3):
            with gateway.acquire("coder"):
                pass
        self.assertEqual(gateway.stats()["switches"], 0)

    def test_what_is_on_the_card_is_forgotten_when_it_is_taken_off(self):
        """The page can stop a model, and then the card is not what it was.

        This used to be answered by reading the world on every call, which cost
        the expensive question each time the page refreshed. The routes behind
        those buttons say so instead, which is cheaper and cannot be stale for
        anything this manager did itself.
        """
        operations = two_models()
        gateway = quick(operations)
        with gateway.acquire("coder"):
            pass
        self.assertEqual(gateway.stats()["current"], "coder")

        operations.unload("coder")              # straight past the gateway
        gateway.card_changed()                  # which the routes report
        self.assertIsNone(gateway.stats()["current"])

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

    def test_handing_a_place_back_twice_does_not_free_somebody_else_s(self):
        # A request that fails while being forwarded gives its place back
        # twice: once from the code that noticed, once from the reader's
        # cleanup. Each lease keeps its own answer to "have I given it back",
        # because several are held at once now and one flag could not.
        operations = two_models(coder=True)
        gateway = quick(operations)
        lease = gateway.acquire("coder")
        lease.release()
        lease.release()
        self.assertEqual(gateway.stats()["in_flight"], 0)

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

    A card found with two engines up is not a card to adopt, it is a card to
    clear. The gateway only takes up what it finds when it finds exactly one
    model answering; anything else is left unknown, and the next request
    switches — which unloads everything before it loads.
    """

    def test_a_stray_is_unloaded(self):
        # Both are up: a manager restart found two units enabled, or somebody
        # pressed Load twice. The request asks for one of them.
        operations = two_models(coder=True, reviewer=True)
        gateway = quick(operations)

        with gateway.acquire("coder") as lease:
            self.assertEqual(lease.port, 8080)

        self.assertIn("reviewer", operations.unloads)
        running = [e["id"] for e in operations.instances() if e["running"]]
        self.assertEqual(running, ["coder"])

    def test_a_card_with_one_model_answering_is_taken_up_as_it_is(self):
        # The ordinary case on Linux: systemd keeps the engines across a
        # manager restart. Reloading would be a minute of work to arrive where
        # it already was, and would take the card from whoever was using it.
        operations = two_models(coder=True)
        gateway = quick(operations)
        with gateway.acquire("coder"):
            pass
        self.assertEqual(operations.loads, [])
        self.assertEqual(operations.unloads, [])

    def test_a_clean_card_is_left_alone(self):
        operations = two_models(coder=True)
        gateway = quick(operations)
        with gateway.acquire("coder"):
            pass
        self.assertEqual(operations.unloads, [])
        self.assertEqual(gateway.stats()["recent"], [])

    def test_the_card_is_read_once_and_then_believed(self):
        # After the first look, the scheduler is the authority. Reading again
        # on every request is the expensive question, and the answer is one it
        # already knows.
        operations = two_models(coder=True)
        gateway = quick(operations)
        with gateway.acquire("coder"):
            pass
        operations._entries["reviewer"].update(running=True, ready=True)
        with gateway.acquire("coder"):
            pass
        self.assertEqual(operations.unloads, [],
                         "it went looking again when it had been told")


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

        def refuse(instance_id, settings=None):
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

    def test_a_warm_request_never_asks_the_supervisor(self):
        """The expensive question is not asked on the ordinary path at all.

        It was three reads, then two, then one, and now none. Everything the
        front door needs to route a request — which entry answers to a name,
        which engine runs it, what settings it has — is the configuration file,
        at 0.05 ms. What every instance is *doing* costs 73 ms on the container,
        and that is only needed when something outside may have moved a model.
        """
        operations = self.counting(coder=True)
        gateway = quick(operations)
        with gateway.acquire("coder"):
            pass                                 # the first look adopts the card
        before = operations.reads
        with gateway.acquire("coder"):
            pass
        self.assertEqual(operations.reads, before,
                         "it asked the supervisor for a request that needed nothing")
        self.assertGreater(operations.cheap_reads, 0)

    def test_the_first_request_looks_once_so_it_can_take_up_what_is_there(self):
        # systemd keeps engines across a manager restart, so a manager coming
        # back finds its model still answering. One look, then it is believed.
        operations = self.counting(coder=True)
        gateway = quick(operations)
        with gateway.acquire("coder"):
            pass
        self.assertEqual(operations.reads, 1)
        self.assertEqual(operations.loads, [], "it reloaded what was already up")

    def test_the_engine_name_costs_no_extra_read(self):
        # It is pure configuration, and the lease carries it. Asking for it
        # afterwards meant asking the expensive question again for an answer
        # that was already in hand.
        operations = self.counting(coder=True)
        gateway = quick(operations)
        with gateway.acquire("coder") as lease:
            self.assertEqual(lease.model_name, "Qwen3.6-35B")
        self.assertEqual(operations.reads, 1)

    def test_how_many_places_a_shape_has_is_a_cheap_question(self):
        # Asked on every admission. It is configuration — how many slots the
        # engine was started with — so it must not cost the expensive read.
        operations = self.counting(coder=True)
        gateway = quick(operations)
        with gateway.acquire("coder"):
            pass
        self.assertEqual(operations.reads, 1)
        self.assertGreater(operations.cheap_reads, 0)


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


class StartupSettingsTests(unittest.TestCase):
    """A request asking for the model started a particular way.

    Some settings go in a request; others decide how the process starts, and
    those cannot. Context size is the one that matters in practice: an agent
    needs room for its own instructions, and the entry may be configured for
    less. Sending it in the body would reach the engine, which does not know
    it and would ignore it without a word.

    So it travels in a field of ours. If the model is already running that way,
    nothing happens. If not, it is reloaded — which never interrupts anything,
    because the card is taken first and only handed over after the request in
    front has had its last byte.
    """

    def test_a_stopped_model_starts_with_what_was_asked_for(self):
        operations = two_models()
        gateway = quick(operations)
        with gateway.acquire("coder", settings={"context_size": 65536}):
            pass
        self.assertEqual(operations.load_settings[-1]["context_size"], 65536)

    def test_a_model_already_running_that_way_is_left_alone(self):
        # Reloading for settings it already has would cost the wait for
        # nothing, on every request.
        operations = two_models(coder=True)
        gateway = quick(operations)
        with gateway.acquire("coder", settings={"context_size": 65536}):
            pass
        self.assertEqual(len(operations.loads), 1)
        with gateway.acquire("coder", settings={"context_size": 65536}):
            pass
        self.assertEqual(len(operations.loads), 1, "it was reloaded for nothing")

    def test_asking_for_something_else_reloads_it(self):
        operations = two_models(coder=True)
        gateway = quick(operations)
        with gateway.acquire("coder", settings={"context_size": 65536}):
            pass
        with gateway.acquire("coder", settings={"context_size": 98304}):
            pass
        self.assertEqual(len(operations.loads), 2)
        self.assertEqual(operations.load_settings[-1]["context_size"], 98304)

    def test_asking_for_what_it_is_configured_with_changes_nothing(self):
        operations = two_models(coder=True)
        gateway = quick(operations)
        with gateway.acquire("coder", settings={"context_size": 32768}):
            pass
        self.assertEqual(operations.loads, [])

    def test_settings_are_not_saved_to_the_entry(self):
        # One request must not quietly rewrite what somebody chose in the page.
        operations = two_models(coder=True)
        gateway = quick(operations)
        with gateway.acquire("coder", settings={"context_size": 65536}):
            pass
        stored = next(item["params"] for item in operations.instances()
                      if item["id"] == "coder")
        self.assertEqual(stored["context_size"], 32768)

    def test_the_difference_is_visible_afterwards(self):
        # Otherwise the page shows a configured value the running model is not
        # using, and there is no way to tell.
        operations = two_models(coder=True)
        gateway = quick(operations)
        with gateway.acquire("coder", settings={"context_size": 65536}):
            pass
        row = next(item for item in operations.instances() if item["id"] == "coder")
        self.assertEqual(row["active_params"], {"context_size": 65536})

    def test_a_setting_the_engine_does_not_have_is_refused(self):
        gateway = quick(two_models(coder=True))
        with self.assertRaises(ValueError) as caught:
            gateway.acquire("coder", settings={"gpu_layers": 20})
        self.assertIn("gpu_layers", str(caught.exception))

    def test_a_value_out_of_range_is_refused(self):
        gateway = quick(two_models(coder=True))
        with self.assertRaises(ValueError):
            gateway.acquire("coder", settings={"context_size": 10})

    def test_a_bad_setting_is_refused_before_anything_is_loaded(self):
        # And before queueing: a misspelling should come back at once, not
        # after waiting behind somebody else's answer.
        operations = two_models()
        gateway = quick(operations)
        with self.assertRaises(ValueError):
            gateway.acquire("coder", settings={"nonsense": 1})
        self.assertEqual(operations.loads, [])
        self.assertIsNone(gateway.busy())

    def test_no_settings_means_the_entry_is_used_as_configured(self):
        # The full configured settings, not an empty answer. A request naming
        # none and one naming exactly the configured values are asking for the
        # same thing, and they have to compare equal or every other request
        # reloads for nothing.
        operations = two_models()
        gateway = quick(operations)
        with gateway.acquire("coder"):
            pass
        self.assertEqual(operations.load_settings,
                         [{"context_size": 32768, "parallel": 1}])


def busy_models(**running):
    """Two entries whose engines serve several requests at once."""
    operations = two_models(**running)
    for entry in operations._entries.values():
        entry["params"] = {"context_size": 32768, "parallel": 4}
    return operations


class TogetherTests(unittest.TestCase):
    """Requests to the model on the card do not take turns.

    They used to, and the comment in this file said an agent workflow was a
    sequence so nothing was lost. That stopped being true the moment subagents
    fanned out over one model — the commonest shape there is — and vLLM's whole
    advantage is that it interleaves requests rather than queueing them.
    """

    def test_two_requests_to_the_loaded_model_run_at_once(self):
        gateway = quick(busy_models(coder=True))
        first = gateway.acquire("coder")
        second = gateway.acquire("coder")
        self.assertEqual(gateway.stats()["in_flight"], 2)
        first.release()
        second.release()

    def test_the_number_of_places_is_the_engine_s_own(self):
        gateway = quick(busy_models())          # nothing running
        self.assertEqual(gateway.stats()["places"], 0, "nothing loaded yet")
        with gateway.acquire("coder"):
            self.assertEqual(gateway.stats()["places"], 4)

    def test_one_slot_still_means_one_at_a_time(self):
        # The GGUF entries are configured that way, and llama.cpp divides the
        # context between slots rather than sharing it. Respecting the number
        # is the point: it was chosen.
        operations = two_models(coder=True)          # parallel: 1
        gateway = quick(operations)
        first = gateway.acquire("coder")
        blocked = threading.Thread(
            target=lambda: gateway.acquire("coder").release(), daemon=True)
        blocked.start()
        time.sleep(0.05)
        self.assertEqual(gateway.stats()["in_flight"], 1)
        self.assertEqual(gateway.stats()["waiting"], 1)
        first.release()
        blocked.join(timeout=2)

    def test_a_request_for_another_model_waits_and_closes_the_door(self):
        operations = busy_models(coder=True)
        gateway = quick(operations)
        held = gateway.acquire("coder")
        waiting = threading.Thread(
            target=lambda: gateway.acquire("reviewer").release(), daemon=True)
        waiting.start()
        time.sleep(0.05)
        self.assertEqual(gateway.stats()["waiting"], 1)
        # Now one for the loaded model, which would have walked in a moment ago.
        later = threading.Thread(
            target=lambda: gateway.acquire("coder").release(), daemon=True)
        later.start()
        time.sleep(0.05)
        self.assertEqual(gateway.stats()["waiting"], 2)
        held.release()
        waiting.join(timeout=3)
        later.join(timeout=3)
        self.assertIn("reviewer", operations.loads)

    def test_the_page_can_see_who_is_waiting_and_for_what(self):
        operations = busy_models(coder=True)
        gateway = quick(operations)
        held = gateway.acquire("coder")
        for _ in range(2):
            threading.Thread(
                target=lambda: _ignore(gateway.acquire, "reviewer"),
                daemon=True).start()
        time.sleep(0.05)
        summary = gateway.stats()["waiting_for"]
        self.assertEqual([row["instance_id"] for row in summary], ["reviewer"])
        self.assertEqual(summary[0]["waiting"], 2)
        held.release()

    def test_the_settings_asked_for_split_a_model_into_two_shapes(self):
        # Two requests for one entry wanting different context sizes are not
        # requests for the same thing: one of them needs a reload.
        operations = busy_models(coder=True)
        gateway = quick(operations)
        held = gateway.acquire("coder", settings={"context_size": 65536})
        other = threading.Thread(
            target=lambda: _ignore(gateway.acquire, "coder"), daemon=True)
        other.start()
        time.sleep(0.05)
        self.assertEqual(gateway.stats()["waiting"], 1,
                         "it shared a card with a different context size")
        held.release()


class BusyMessageTests(unittest.TestCase):
    """What the page is told before it interrupts anything."""

    def test_it_counts_the_answers_in_progress(self):
        gateway = quick(busy_models(coder=True))
        gateway.acquire("coder")
        gateway.acquire("coder")
        with self.assertRaises(CardBusy) as caught:
            gateway.guard("unload", "coder")
        self.assertIn("2 requests", str(caught.exception))

    def test_it_counts_the_ones_waiting_too(self):
        # One answer being written is a different thing from forty requests
        # waiting for a model, and the difference decides whether to go ahead.
        gateway = quick(two_models(coder=True))          # one place
        gateway.acquire("coder")
        for _ in range(3):
            threading.Thread(target=lambda: _ignore(gateway.acquire, "reviewer"),
                             daemon=True).start()
        time.sleep(0.05)
        with self.assertRaises(CardBusy) as caught:
            gateway.guard("unload", "coder")
        self.assertIn("3 more are waiting", str(caught.exception))

    def test_an_idle_card_is_not_busy(self):
        gateway = quick(busy_models(coder=True))
        with gateway.acquire("coder"):
            pass
        self.assertIsNone(gateway.busy())
        gateway.guard("unload", "coder")             # does not raise


class ForcedResetTests(unittest.TestCase):
    """A forced stop leaves a clean slate, not a partly true one."""

    def test_everyone_waiting_is_turned_away(self):
        gateway = quick(two_models(coder=True))
        held = gateway.acquire("coder")
        refused = []
        for _ in range(3):
            threading.Thread(
                target=lambda: refused.append(_ignore(gateway.acquire, "reviewer")),
                daemon=True).start()
        time.sleep(0.05)
        self.assertEqual(gateway.reset("stopped by hand"), 3)
        time.sleep(0.05)
        self.assertEqual(len(refused), 3)
        held.release()

    def test_nothing_is_running_or_waiting_afterwards(self):
        operations = busy_models(coder=True)
        gateway = quick(operations)
        gateway.acquire("coder")
        gateway.acquire("coder")
        gateway.reset("stopped by hand")
        # Whoever forced it also stopped the engine, which is what forcing is.
        operations._entries["coder"].update(running=False, ready=False)
        stats = gateway.stats()
        self.assertEqual(stats["in_flight"], 0)
        self.assertEqual(stats["waiting"], 0)
        self.assertIsNone(stats["current"])


def _ignore(function, *args, **kwargs):
    try:
        lease = function(*args, **kwargs)
        lease.release()
        return None
    except Exception as error:
        return error


class LiveSettingsTests(unittest.TestCase):
    """Changing a limit reaches the next request, not the next restart."""

    def test_the_waits_are_read_from_the_gateway_each_time(self):
        gateway = quick(two_models())
        gateway.apply_settings({"first_byte_s": 300, "between_bytes_s": 45})
        self.assertEqual(gateway.first_byte_s, 300.0)
        self.assertEqual(gateway.between_bytes_s, 45.0)
        self.assertEqual(gateway.stats()["first_byte_s"], 300.0)

    def test_the_queue_length_reaches_the_scheduler(self):
        gateway = quick(two_models())
        gateway.apply_settings({"max_waiting": 7})
        self.assertEqual(gateway.scheduler.max_waiting, 7)
        self.assertEqual(gateway.stats()["max_waiting"], 7)

    def test_settings_left_out_are_left_alone(self):
        gateway = quick(two_models())
        before = gateway.between_bytes_s
        gateway.apply_settings({"first_byte_s": 200})
        self.assertEqual(gateway.between_bytes_s, before)


class ShapesOfferedTests(unittest.TestCase):
    """What the page tells a client it may send, and to which models."""

    def test_the_usual_shape_is_answered_by_everything(self):
        gateway = quick(mixed_engines())
        rows = {row["path"]: row["models"] for row in gateway.stats()["shapes"]}
        self.assertEqual(rows["/v1/chat/completions"], ["coder", "fast"])

    def test_the_other_shape_names_only_the_engines_that_read_it(self):
        gateway = quick(mixed_engines())
        rows = {row["path"]: row["models"] for row in gateway.stats()["shapes"]}
        self.assertEqual(rows["/v1/messages"], ["fast"])

    def test_it_costs_only_the_configuration(self):
        # Asked every few seconds by the page. Working out which engine answers
        # what is the configuration; asking the supervisor what every instance
        # is doing would undo the reason the page is cheap.
        operations = mixed_engines()
        reads = {"n": 0}
        original = operations.instances

        def counted():
            reads["n"] += 1
            return original()
        operations.instances = counted
        quick(operations)._shapes_offered()
        self.assertEqual(reads["n"], 0)


class QueueRunsTests(unittest.TestCase):
    """The queue reported as the order it will be served in.

    It is served in order and requests next to each other wanting the same
    model go in together, so it is a list of turns rather than a list of
    requests. Grouped the same way the scheduler groups them, the page shows
    the model changes that are about to happen.
    """

    def queued(self, *shapes):
        operations = busy_models(coder=True)
        gateway = quick(operations)
        held = gateway.acquire("coder")
        for shape in shapes:
            threading.Thread(target=lambda s=shape: _ignore(gateway.acquire, s),
                             daemon=True).start()
            time.sleep(0.03)
        time.sleep(0.05)
        runs = gateway.stats()["queue_runs"]
        held.release()
        return [(run["instance_id"], run["requests"]) for run in runs]

    def test_requests_next_to_each_other_are_one_turn(self):
        self.assertEqual(self.queued("reviewer", "reviewer", "reviewer"),
                         [("reviewer", 3)])

    def test_a_different_model_starts_a_new_turn(self):
        self.assertEqual(self.queued("reviewer", "reviewer", "coder"),
                         [("reviewer", 2), ("coder", 1)])

    def test_the_same_model_after_another_is_a_separate_turn(self):
        # Which is the ordering rule made visible: those two reviewers are not
        # served with the first two, because a request for coder arrived
        # between them.
        self.assertEqual(self.queued("reviewer", "coder", "reviewer"),
                         [("reviewer", 1), ("coder", 1), ("reviewer", 1)])

    def test_an_empty_queue_has_no_turns(self):
        gateway = quick(busy_models(coder=True))
        self.assertEqual(gateway.stats()["queue_runs"], [])


class CardReadingTests(unittest.TestCase):
    """What the accelerator reports, on the page that watches it."""

    def test_memory_is_reported(self):
        card = quick(two_models()).stats()["card"]
        self.assertEqual(card["total_mb"], 32000)
        self.assertIn("used_mb", card)

    def test_a_card_that_cannot_be_read_reports_nothing_rather_than_zero(self):
        # Zero of zero would read as an empty card, which is a different thing
        # from a card nobody could ask.
        operations = two_models()

        def refuse():
            raise RuntimeError("nvidia-smi is not here")
        operations.host.accelerator = refuse
        self.assertEqual(quick(operations).stats()["card"], {})


class WhatThePageSeesTests(unittest.TestCase):
    """The page must not report an empty card that has a model on it.

    Found on the machine: after a deployment the manager restarts, systemd
    keeps the engine running, and the gateway does not look at the card until
    a request arrives — so the page said "nothing loaded" for as long as
    nobody sent anything, which is exactly when somebody is watching it.
    """

    def test_it_reports_a_model_nobody_has_asked_for_yet(self):
        gateway = quick(two_models(coder=True))
        self.assertEqual(gateway.stats()["current"], "coder")

    def test_looking_costs_the_expensive_read_once(self):
        operations = two_models(coder=True)
        reads = {"n": 0}
        original = operations.instances

        def counted():
            reads["n"] += 1
            return original()
        operations.instances = counted
        gateway = quick(operations)
        # The fake accelerator asks what is running to decide what to report,
        # which the real one does not. Counting its reads would count the
        # fake's habits rather than the gateway's.
        operations.host.accelerator = lambda pid=None: _no_card()
        gateway.stats()
        gateway.stats()
        gateway.stats()
        self.assertEqual(reads["n"], 1, "it looked again on every refresh")

    def test_it_looks_again_after_a_button_moves_a_model(self):
        operations = two_models(coder=True)
        gateway = quick(operations)
        self.assertEqual(gateway.stats()["current"], "coder")
        operations.unload("coder")
        operations._entries["reviewer"].update(running=True, ready=True)
        gateway.card_changed()
        self.assertEqual(gateway.stats()["current"], "reviewer")

    def test_two_models_up_is_not_a_card_to_report(self):
        # It is a card to clear, and the next request clears it.
        gateway = quick(two_models(coder=True, reviewer=True))
        self.assertIsNone(gateway.stats()["current"])


def _no_card():
    from ai_lab.types import AcceleratorSnapshot
    return AcceleratorSnapshot(available=True, name="Fake", kind="cuda",
                               memory_kind="dedicated",
                               memory_used_mb=2.0, memory_total_mb=32000.0)
