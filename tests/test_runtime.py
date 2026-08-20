import time
import unittest

from ai_lab.config import Instance
from ai_lab.events import EventBus
from ai_lab.runtime import Runtime
from ai_lab.types import Format, ModelFile, ModelSet, Phase, RuntimeEvent

from tests.support import FakeEngine, FakeHost


def instance(identifier="qwen", port=8080):
    return Instance(id=identifier, name="Coding", engine="fake",
                    model_id="repo/qwen", port=port)


def model(name="qwen"):
    return ModelSet(id=f"repo/{name}", name=name, format=Format.GGUF,
                    entrypoint=f"/models/{name}.gguf",
                    files=(ModelFile(f"/models/{name}.gguf", 100),))


class LoadTests(unittest.TestCase):
    def setUp(self):
        self.bus = EventBus()
        self.subscription = self.bus.subscribe()

    def runtime(self, host):
        return Runtime(host, self.bus, sample_interval_s=0)

    def phases(self, operation):
        return [step.phase for step in operation.steps]

    def _load(self, host, engine=None, model_name="qwen"):
        engine = engine or FakeEngine(host=host)
        return self.runtime(host).load(instance(), model(model_name), engine)

    def test_a_load_walks_every_phase_in_order(self):
        operation = self._load(FakeHost())
        self.assertTrue(operation.ok)
        self.assertEqual(self.phases(operation),
                         [Phase.STARTING, Phase.PROCESS_UP, Phase.READY])

    def test_each_phase_carries_a_time(self):
        operation = self._load(FakeHost())
        times = [step.elapsed_ms for step in operation.steps]
        self.assertEqual(times, sorted(times))
        self.assertGreaterEqual(operation.total_ms, times[-1])

    def test_the_engine_command_reaches_the_host(self):
        host = FakeHost()
        self._load(host)
        self.assertEqual(len(host.started), 1)
        self.assertIn("/models/qwen.gguf", host.started[0].argv)
        self.assertEqual(host.started[0].instance_id, "qwen")

    def test_ready_is_awaited_not_assumed(self):
        """The process being up is not the same as the weights being loaded."""
        host = FakeHost()
        engine = FakeEngine(ready_after=4, host=host)
        operation = self.runtime(host).load(instance(), model(), engine)
        self.assertTrue(operation.ok)
        self.assertGreaterEqual(engine.probes, 4)

    def test_a_failing_plan_is_reported_not_raised(self):
        class Broken(FakeEngine):
            def plan(self, model, port, params):
                raise ValueError("model is missing a shard")

        operation = self.runtime(FakeHost()).load(instance(), model(), Broken())
        self.assertFalse(operation.ok)
        self.assertEqual(operation.error, "model is missing a shard")
        self.assertEqual(self.phases(operation)[-1], Phase.FAILED)


class UnloadTests(unittest.TestCase):
    def setUp(self):
        self.bus = EventBus()

    def test_an_unload_waits_for_memory_to_settle(self):
        """A process exiting and its memory coming back are different moments."""
        host = FakeHost(memory_curve=[20000, 20000, 12000, 4000, 400, 400, 400, 400])
        host.running.add("qwen")
        runtime = Runtime(host, self.bus, sample_interval_s=0)
        operation = runtime.unload("qwen")
        self.assertTrue(operation.ok)
        self.assertEqual([step.phase for step in operation.steps],
                         [Phase.STOPPING, Phase.PROCESS_GONE, Phase.MEMORY_RELEASED])
        self.assertEqual(host.stopped, ["qwen"])


class SwapTests(unittest.TestCase):
    def setUp(self):
        self.bus = EventBus()

    def test_a_swap_unloads_then_loads_and_reports_one_total(self):
        host = FakeHost(memory_curve=[20000, 1000, 1000, 1000, 1000, 8000, 16000])
        host.running.add("qwen")
        operation = Runtime(host, self.bus, sample_interval_s=0).swap(
            instance(), model("gemma"), FakeEngine(host=host))
        self.assertTrue(operation.ok)
        self.assertEqual(operation.kind, "swap")
        phases = [step.phase for step in operation.steps]
        self.assertEqual(phases[0], Phase.STOPPING)
        self.assertEqual(phases[-1], Phase.READY)
        self.assertEqual(host.stopped, ["qwen"])
        self.assertEqual(len(host.started), 1)

    def test_swapping_into_an_idle_slot_skips_the_unload(self):
        host = FakeHost()
        operation = Runtime(host, self.bus, sample_interval_s=0).swap(
            instance(), model(), FakeEngine(host=host))
        self.assertTrue(operation.ok)
        self.assertEqual(host.stopped, [])
        self.assertEqual([step.phase for step in operation.steps][0], Phase.STARTING)


class EventTests(unittest.TestCase):
    def test_memory_readings_are_published_while_loading(self):
        bus = EventBus()
        subscription = bus.subscribe()
        host = FakeHost(memory_curve=[0, 4000, 12000, 22000])
        Runtime(host, bus, sample_interval_s=0).load(
            instance(), model(), FakeEngine(ready_after=3, host=host))
        stream = subscription.events(timeout=0.01)
        readings = []
        while True:
            event = next(stream)
            if event is None:
                break
            if isinstance(event, RuntimeEvent):
                readings.append(event.memory_used_mb)
        self.assertGreater(len(readings), 3)
        self.assertGreater(max(readings), 0)

    def test_the_last_operation_is_remembered(self):
        bus = EventBus()
        host = FakeHost()
        runtime = Runtime(host, bus, sample_interval_s=0)
        self.assertIsNone(runtime.last("qwen"))
        runtime.load(instance(), model(), FakeEngine(host=host))
        self.assertEqual(runtime.last("qwen")["kind"], "load")
        self.assertTrue(runtime.last("qwen")["ok"])


if __name__ == "__main__":
    unittest.main()


class StrangerOnThePortTests(unittest.TestCase):
    """Something else answering the port must not be reported as a load.

    Found in real use: an engine left behind by a previous manager kept its
    port, the readiness probe said yes, and a load that never happened was
    reported as a 42 ms success.
    """

    def setUp(self):
        self.bus = EventBus()

    def test_a_load_over_a_stranger_is_refused(self):
        class AlwaysReady(FakeEngine):
            def ready(self, port):
                return True

        host = FakeHost()
        operation = Runtime(host, self.bus, sample_interval_s=0).load(
            instance(), model(), AlwaysReady())
        self.assertFalse(operation.ok)
        self.assertIn("already answering", operation.error)
        self.assertEqual(host.started, [])

    def test_our_own_running_instance_is_not_mistaken_for_a_stranger(self):
        host = FakeHost()
        host.running.add("qwen")
        operation = Runtime(host, self.bus, sample_interval_s=0).load(
            instance(), model(), FakeEngine(host=host))
        self.assertTrue(operation.ok, operation.error)


class PerInstanceMemoryTests(unittest.TestCase):
    """The bar must follow one model, not the whole card.

    Found in use: unloading Gemma while Qwen stayed resident left Gemma's bar
    sitting at 64%, because the readings were the card total and Qwen still
    held 21 GB of it.
    """

    def setUp(self):
        self.bus = EventBus()
        self.subscription = self.bus.subscribe()

    def drain(self):
        """Only the progress events; the stream also carries change notices."""
        stream = self.subscription.events(timeout=0.01)
        events = []
        while (event := next(stream)) is not None:
            if isinstance(event, RuntimeEvent):
                events.append(event)
        return events

    def test_unloading_one_model_ignores_what_another_still_holds(self):
        host = FakeHost(memory_curve=[3500, 0, 0, 0, 0, 0], other_models_mb=21000)
        host.running.add("qwen")
        operation = Runtime(host, self.bus, sample_interval_s=0).unload("qwen")
        self.assertTrue(operation.ok)
        final = self.drain()[-1]
        self.assertEqual(final.memory_used_mb, 0.0)          # this model: gone
        self.assertEqual(final.accelerator_used_mb, 21000)   # the other one: still there

    def test_a_load_reports_this_instance_rising(self):
        host = FakeHost(memory_curve=[0, 1000, 3500], other_models_mb=21000)
        operation = Runtime(host, self.bus, sample_interval_s=0).load(
            instance(), model(), FakeEngine(ready_after=3, host=host))
        self.assertTrue(operation.ok, operation.error)
        readings = [event.memory_used_mb for event in self.drain()]
        self.assertEqual(readings[0], 0.0)                   # before the process exists
        self.assertGreater(max(readings), 0)

    def test_the_whole_card_is_still_reported_for_context(self):
        host = FakeHost(memory_curve=[3500], other_models_mb=21000)
        host.running.add("qwen")
        Runtime(host, self.bus, sample_interval_s=0).unload("qwen")
        self.assertTrue(all(event.accelerator_used_mb >= 21000 for event in self.drain()))


class ModelThatDoesNotFitTests(unittest.TestCase):
    """A model too large for the free memory must fail quickly and explain why.

    Found in use: llama-server exited after two seconds with a clear message in
    its own log, and the manager sat waiting out its fifteen-minute timeout.
    """

    def setUp(self):
        self.bus = EventBus()

    def test_refused_before_anything_starts_when_the_weights_cannot_fit(self):
        host = FakeHost(total_mb=32000, other_models_mb=28000)
        big = ModelSet(id="r/big", name="big", format=Format.GGUF,
                       entrypoint="/models/big.gguf",
                       files=(ModelFile("/models/big.gguf", 20 * 1024**3),))
        operation = Runtime(host, self.bus, sample_interval_s=0).load(
            instance(), big, FakeEngine(host=host))
        self.assertFalse(operation.ok)
        self.assertIn("only", operation.error)
        self.assertIn("free on the card", operation.error)
        self.assertEqual(host.started, [], "nothing should have been started")

    def test_the_refusal_offers_the_split_as_a_way_out(self):
        host = FakeHost(total_mb=32000, other_models_mb=28000)
        operation = Runtime(host, self.bus, sample_interval_s=0).load(
            instance(), self.big(), FakeEngine(host=host))
        self.assertFalse(operation.ok)
        self.assertIn("choose a smaller one", operation.error)
        self.assertIn("system memory", operation.error)

    def test_a_deliberate_split_is_not_refused(self):
        """A model larger than the card runs when you asked for it to be split.

        It is slow — most of the weights sit in system memory and are reached
        over the link — but it is a choice, and refusing it would be answering
        a question nobody asked.
        """
        host = FakeHost(total_mb=32000, other_models_mb=28000)
        operation = Runtime(host, self.bus, sample_interval_s=0).load(
            instance(), self.big(),
            FakeEngine(host=host, splits_across_cpu=True))
        self.assertTrue(operation.ok, operation.error)
        self.assertEqual(len(host.started), 1)

    @staticmethod
    def big():
        return ModelSet(id="r/big", name="big", format=Format.GGUF,
                        entrypoint="/models/big.gguf",
                        files=(ModelFile("/models/big.gguf", 20 * 1024**3),))

    def test_an_engine_that_dies_mid_load_fails_at_once(self):
        class DiesOnStart(FakeHost):
            def start(self, spec):
                super().start(spec)
                self.running.discard(spec.instance_id)   # exits immediately

        host = DiesOnStart()
        host.log_lines = ["load_model: failed to load model, out of memory"]
        started = time.monotonic()
        runtime = Runtime(host, self.bus, sample_interval_s=0,
                          start_timeout_s=0.5, load_timeout_s=0.5)
        operation = runtime.load(instance(), model(), FakeEngine(host=host))
        self.assertFalse(operation.ok)
        self.assertLess(time.monotonic() - started, 5, "should not wait out the timeout")
        self.assertIn("out of memory", operation.error)

    def test_the_engines_own_error_line_is_reported(self):
        class DiesAfterStart(FakeHost):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self._checks = 0

            def status(self, instance_id):
                self._checks += 1
                if self._checks > 2:
                    self.running.discard(instance_id)
                return super().status(instance_id)

        host = DiesAfterStart()
        host.log_lines = ["ggml_backend_cuda_buffer_type_alloc_buffer: "
                          "allocating 20480.00 MiB on device 0: cudaMalloc failed: out of memory"]
        operation = Runtime(host, self.bus, sample_interval_s=0).load(
            instance(), model(), FakeEngine(ready_after=99, host=host))
        self.assertFalse(operation.ok)
        self.assertIn("out of memory", operation.error)


class TimeoutTests(unittest.TestCase):
    """Starting a process and loading weights get separate limits."""

    def test_a_process_that_never_appears_gives_up_on_the_start_limit(self):
        class NeverStarts(FakeHost):
            def start(self, spec):
                self.started.append(spec)      # accepted, but never runs

        host = NeverStarts()
        host.log_lines = ["unit failed to execute: no such file"]
        started = time.monotonic()
        operation = Runtime(host, EventBus(), sample_interval_s=0,
                            start_timeout_s=0.3).load(
            instance(), model(), FakeEngine(host=host))
        self.assertFalse(operation.ok)
        self.assertLess(time.monotonic() - started, 3)
        self.assertIn("no such file", operation.error)


class FailureMessageTests(unittest.TestCase):
    """The message must quote the engine, not the supervisor.

    systemd's own summary lines come last in the journal and say only that a
    process exited, which is exactly what the user already knows.
    """

    def load_with_logs(self, log_lines):
        class DiesAfterStart(FakeHost):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self._checks = 0

            def status(self, instance_id):
                self._checks += 1
                if self._checks > 2:
                    self.running.discard(instance_id)
                return super().status(instance_id)

        host = DiesAfterStart()
        host.log_lines = log_lines
        return Runtime(host, EventBus(), sample_interval_s=0).load(
            instance(), model(), FakeEngine(ready_after=99, host=host))

    def test_the_supervisors_summary_is_skipped(self):
        operation = self.load_with_logs([
            "llama_model_load: error loading model: unable to allocate buffer",
            "ai-lab-engine@qwen.service: Main process exited, code=exited",
            "ai-lab-engine@qwen.service: Failed with result 'exit-code'.",
        ])
        self.assertIn("unable to allocate buffer", operation.error)
        self.assertNotIn("exit-code", operation.error)

    def test_an_unreadable_journal_says_how_to_fix_it(self):
        operation = self.load_with_logs([])
        self.assertIn("systemd-journal", operation.error)


class ProgressBarTests(unittest.TestCase):
    """The bar reports how far the operation has got, not how full the card is.

    Memory occupancy makes a poor bar: a 4 GB model on a 32 GB card would show
    13% while being completely loaded.
    """

    def setUp(self):
        self.bus = EventBus()
        self.subscription = self.bus.subscribe()

    def drain(self):
        """Only the progress events; the stream also carries change notices."""
        stream = self.subscription.events(timeout=0.01)
        events = []
        while (event := next(stream)) is not None:
            if isinstance(event, RuntimeEvent):
                events.append(event)
        return events

    def four_gb_model(self):
        return ModelSet(id="r/m", name="m", format=Format.GGUF,
                        entrypoint="/models/m.gguf",
                        files=(ModelFile("/models/m.gguf", 4 * 1024**3),))

    def test_a_load_runs_from_nearly_zero_to_one(self):
        host = FakeHost(memory_curve=[0, 1024, 2048, 4096], total_mb=32000)
        operation = Runtime(host, self.bus, sample_interval_s=0).load(
            instance(), self.four_gb_model(), FakeEngine(ready_after=4, host=host))
        self.assertTrue(operation.ok, operation.error)
        values = [event.progress for event in self.drain()]
        self.assertLess(values[0], 0.1)
        self.assertEqual(values[-1], 1.0)
        self.assertEqual(values, sorted(values), "progress must not go backwards")

    def test_a_small_model_still_reaches_a_hundred_percent(self):
        """The point of the change: occupancy would have stopped at 13%."""
        host = FakeHost(memory_curve=[0, 4096], total_mb=32000)
        Runtime(host, self.bus, sample_interval_s=0).load(
            instance(), self.four_gb_model(), FakeEngine(ready_after=2, host=host))
        self.assertEqual(self.drain()[-1].progress, 1.0)

    def test_an_unload_also_runs_up_to_one_as_memory_falls(self):
        host = FakeHost(memory_curve=[4096, 2048, 0, 0, 0, 0], total_mb=32000)
        host.running.add("qwen")
        operation = Runtime(host, self.bus, sample_interval_s=0).unload("qwen")
        self.assertTrue(operation.ok)
        values = [event.progress for event in self.drain()]
        self.assertEqual(values[-1], 1.0)
        self.assertEqual(values, sorted(values))

    def test_a_swap_is_one_bar_across_both_halves(self):
        host = FakeHost(memory_curve=[4096, 0, 0, 0, 0, 0, 2048, 4096], total_mb=32000)
        host.running.add("qwen")
        operation = Runtime(host, self.bus, sample_interval_s=0).swap(
            instance(), self.four_gb_model(), FakeEngine(ready_after=8, host=host))
        self.assertTrue(operation.ok, operation.error)
        values = [event.progress for event in self.drain()]
        self.assertEqual(values[-1], 1.0)
        self.assertEqual(values, sorted(values), "one continuous bar, not two")

    def test_a_failure_leaves_the_bar_where_it_stopped(self):
        class DiesAfterStart(FakeHost):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self._checks = 0

            def status(self, instance_id):
                self._checks += 1
                if self._checks > 2:
                    self.running.discard(instance_id)
                return super().status(instance_id)

        host = DiesAfterStart(total_mb=32000)
        host.log_lines = ["error loading model"]
        operation = Runtime(host, self.bus, sample_interval_s=0).load(
            instance(), self.four_gb_model(), FakeEngine(ready_after=99, host=host))
        self.assertFalse(operation.ok)
        self.assertLess(self.drain()[-1].progress, 1.0)


class ChangeNoticeTests(unittest.TestCase):
    """The page is told when something moved, instead of asking every few seconds.

    Polling redrew the page whether or not anything had happened, which threw
    away whatever was half-typed or selected at the time.
    """

    def setUp(self):
        self.bus = EventBus()
        self.subscription = self.bus.subscribe()

    def notices(self):
        from ai_lab.types import ChangeEvent
        stream = self.subscription.events(timeout=0.01)
        topics = []
        while (event := next(stream)) is not None:
            if isinstance(event, ChangeEvent):
                topics.append(event.topic)
        return topics

    def test_a_finished_load_says_the_instances_moved(self):
        host = FakeHost()
        Runtime(host, self.bus, sample_interval_s=0).load(
            instance(), model(), FakeEngine(host=host))
        self.assertIn("instances", self.notices())

    def test_a_failed_load_says_so_too(self):
        """A failure changes the list as much as a success does."""
        class Broken(FakeEngine):
            def plan(self, model, port, params):
                raise ValueError("nope")

        Runtime(FakeHost(), self.bus, sample_interval_s=0).load(
            instance(), model(), Broken())
        self.assertIn("instances", self.notices())

    def test_an_unload_says_so(self):
        host = FakeHost()
        host.running.add("qwen")
        Runtime(host, self.bus, sample_interval_s=0).unload("qwen")
        self.assertIn("instances", self.notices())
