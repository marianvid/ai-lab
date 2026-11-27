import json
import threading
import unittest
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory

from ai_lab.providers import ProviderCallCancelledError
from ai_lab.providers.gate import VendorGate, VendorPausedError
from ai_lab.providers.settings import VendorLimits
from ai_lab.providers.usage import DAYS_KEPT, FILE_NAME, UsageLog


class FakeClock:
    def __init__(self):
        self.now = 1000.0
        self.slept = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.slept.append(round(seconds, 3))
        self.now += seconds


class VendorGateTests(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.gate = VendorGate(VendorLimits(concurrency=2, launch_spacing_s=0.5,
                                            rate_limit_waits_s=(30, 120, 300)),
                               clock=self.clock, sleep=self.clock.sleep)

    def test_launches_are_spaced(self):
        for _ in range(3):
            with self.gate.slot():
                pass
        self.assertEqual(self.clock.slept, [0.5, 0.5])

    def test_a_rate_limit_pauses_the_vendor_longer_each_time(self):
        self.assertEqual([self.gate.rate_limited() for _ in range(4)], [30, 120, 300, 300])
        with self.gate.slot():
            pass
        self.assertEqual(self.clock.slept, [300])

    def test_a_success_starts_the_count_again(self):
        self.gate.rate_limited()
        self.gate.succeeded()
        self.assertEqual(self.gate.rate_limited(), 30)

    def test_state_reports_places_and_pause(self):
        self.gate.rate_limited()
        with self.gate.slot():
            pass
        self.clock.now -= 10
        state = self.gate.state()
        self.assertEqual((state["in_flight"], state["concurrency"]), (0, 2))
        self.assertEqual(state["paused_for_s"], 10.0)
        self.assertEqual(state["rate_limit_strikes"], 1)

    def test_a_call_whose_client_leaves_during_a_pause_never_starts(self):
        self.gate.rate_limited()  # 30 s pause
        answers = iter([True, True, True, False])
        with self.assertRaises(ProviderCallCancelledError), self.gate.slot(lambda: next(answers)):
            self.fail("the call started")
        # It looked once a second, not after the whole pause.
        self.assertEqual(self.clock.slept, [1.0, 1.0, 1.0])
        self.assertEqual(self.gate.state()["in_flight"], 0)
        # Its place was given back: two more calls get in.
        with self.gate.slot(), self.gate.slot():
            pass

    def test_a_call_whose_client_leaves_while_the_places_are_full_never_starts(self):
        gate = VendorGate(VendorLimits(concurrency=1, launch_spacing_s=0, client_check_s=0.05))
        answers = iter([True, False])
        with gate.slot(), self.assertRaises(ProviderCallCancelledError), \
                gate.slot(lambda: next(answers)):
            self.fail("the call started")
        with gate.slot():
            pass

    def test_a_call_that_may_not_be_held_leaves_at_a_pause(self):
        self.gate.rate_limited()  # 30 s pause
        with self.assertRaises(VendorPausedError) as caught, self.gate.slot(hold=False):
            self.fail("the call started")
        self.assertEqual(caught.exception.retry_in_s, 30)
        self.assertEqual(self.clock.slept, [])

    def test_a_call_that_may_not_be_held_waits_for_a_place_but_not_for_a_pause(self):
        gate = VendorGate(VendorLimits(concurrency=1, launch_spacing_s=0, client_check_s=0.05,
                                       rate_limit_waits_s=(300,)))
        with gate.slot():
            # Only a full place: it waits. The vendor is paused meanwhile: it leaves.
            def pause_soon():
                gate.rate_limited()
            timer = threading.Timer(0.2, pause_soon)
            timer.start()
            with self.assertRaises(VendorPausedError), gate.slot(hold=False):
                self.fail("the call started")
            timer.join()

    def test_no_more_than_the_concurrency_is_in_flight(self):
        gate = VendorGate(VendorLimits(concurrency=2, launch_spacing_s=0))
        inside, peak, lock = [0], [0], threading.Lock()
        release = threading.Event()

        def call():
            with gate.slot():
                with lock:
                    inside[0] += 1
                    peak[0] = max(peak[0], inside[0])
                release.wait(1)
                with lock:
                    inside[0] -= 1

        threads = [threading.Thread(target=call) for _ in range(5)]
        for thread in threads:
            thread.start()
        release.set()
        for thread in threads:
            thread.join()
        self.assertEqual(peak[0], 2)


class UsageLogTests(unittest.TestCase):
    def setUp(self):
        self._temporary = TemporaryDirectory()
        self.addCleanup(self._temporary.cleanup)
        self.state = Path(self._temporary.name)
        self.day = [date(2026, 10, 7)]

    def log(self):
        return UsageLog(self.state, today=lambda: self.day[0])

    def test_calls_and_failures_are_counted_and_survive_a_restart(self):
        self.log().record("anthropic", "sonnet")
        self.log().record("anthropic", "opus", failure="usage limit")
        today = self.log().today()["anthropic"]
        self.assertEqual((today["requests"], today["ok"]), (2, 1))
        self.assertEqual(today["models"], {"sonnet": 1, "opus": 1})
        self.assertEqual(today["failures"], {"usage limit": 1})

    def test_old_days_are_dropped(self):
        log = self.log()
        for offset in range(DAYS_KEPT + 3):
            self.day[0] = date.fromordinal(date(2026, 1, 1).toordinal() + offset)
            log.record("openai", "terra")
        self.assertEqual(len(json.loads((self.state / FILE_NAME).read_text())), DAYS_KEPT)

    def test_a_disk_that_refuses_the_write_does_not_fail_the_call(self):
        # Audit L39: the answer was already given; counting it is best effort.
        log = self.log()
        (self.state / FILE_NAME).mkdir()
        log.record("anthropic", "sonnet")
        self.assertEqual(log.today()["anthropic"]["ok"], 1)

    def test_a_damaged_file_starts_empty(self):
        (self.state / FILE_NAME).write_text("{not json")
        self.assertEqual(self.log().today(), {})


if __name__ == "__main__":
    unittest.main()
