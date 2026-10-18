"""Card first, processor otherwise: where each request goes, and stepping aside."""

import threading
import unittest
from typing import Any, cast

from ai_lab.cpu_models.placement import DRAIN_LIMIT_S, HOLD_LIMIT_S, RECHECK_S, Placement
from ai_lab.cpu_models.settings import CardUse, CpuModel

MODEL = CpuModel(name="bge-m3", path="/m/bge.gguf", port=8390, idle_stop_s=1800.0,
                 card=CardUse(port=8391, needs_mb=1300))


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class Server:
    def __init__(self, device, clock, refuses=False):
        self.device = device
        self.model = MODEL
        self.clock = clock
        self.refuses = refuses
        self.running = False
        self.last_used = 0.0
        self.starts = 0

    def ensure_started(self):
        if self.refuses:
            raise RuntimeError("out of memory")
        if not self.running:
            self.starts += 1
        self.running = True
        self.last_used = self.clock()

    def touch(self):
        self.last_used = self.clock()

    def stop(self):
        self.running = False

    def stop_if_idle(self, busy):
        if busy or not self.running or self.clock() - self.last_used < 1800:
            return False
        self.stop()
        return True


class PlacementTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.free: float | None = 4000.0
        self.reads = 0
        self.cpu = Server("cpu", self.clock)
        self.card = Server("card", self.clock)
        self.place = Placement(cast(Any, self.cpu), cast(Any, self.card), self.card_free,
                               self.clock, self.clock.sleep)

    def card_free(self):
        self.reads += 1
        return self.free

    def test_with_room_a_request_answers_on_the_card(self):
        server = self.place.take()
        self.assertIs(server, self.card)
        self.assertEqual((self.place.on_card(), self.place.device()), (1, "card"))
        self.place.give_back(server)
        self.assertEqual(self.place.on_card(), 0)
        self.assertEqual(self.place.reclaimable_mb(), 1300.0)
        self.assertFalse(self.cpu.running)

    def test_without_room_it_answers_on_the_processor_and_asks_again_later(self):
        self.free = 1700.0                       # needs 1300 + 500
        self.assertIs(self.place.take(), self.cpu)
        self.assertIs(self.place.take(), self.cpu)
        self.assertEqual(self.reads, 1)          # the "no" is trusted for a while
        self.free = 9000.0
        self.clock.now += RECHECK_S + 1
        self.assertIs(self.place.take(), self.card)

    def test_no_card_or_a_refusal_means_the_processor(self):
        self.free = None
        self.assertIs(self.place.take(), self.cpu)
        place = Placement(cast(Any, self.cpu), cast(Any, Server("card", self.clock, refuses=True)),
                          lambda: 9000.0, self.clock, self.clock.sleep)
        self.assertIs(place.take(), self.cpu)
        alone = Placement(cast(Any, self.cpu), None, lambda: 9000.0, self.clock, self.clock.sleep)
        self.assertIs(alone.take(), self.cpu)
        alone.make_way()
        alone.settled()
        self.assertEqual((alone.reclaimable_mb(), alone.device()), (0.0, "cpu"))

    def test_making_way_waits_for_the_card_requests_then_leaves(self):
        finished: list[bool] = []

        def finish_later(seconds):
            self.clock.now += seconds
            if not finished:
                finished.append(True)
                self.place.give_back(server)
        self.place = Placement(cast(Any, self.cpu), cast(Any, self.card), self.card_free,
                               self.clock, finish_later)
        server = self.place.take()
        self.place.make_way()
        self.assertEqual(finished, [True])
        self.assertFalse(self.card.running)
        self.assertEqual(self.place.reclaimable_mb(), 0.0)
        self.assertIs(self.place.take(), self.cpu)   # held while the card model loads

    def test_a_request_that_never_finishes_does_not_block_the_card(self):
        self.place.take()
        self.place.make_way()
        self.assertGreaterEqual(self.clock.now, 1000.0 + DRAIN_LIMIT_S)
        self.assertFalse(self.card.running)

    def test_after_the_load_it_comes_back_if_it_was_in_use(self):
        self.place.give_back(self.place.take())
        self.place.make_way()
        self.place.settled()
        for _ in range(200):
            if self.card.running:
                break
            threading.Event().wait(0.01)
        self.assertTrue(self.card.running)

    def test_after_the_load_it_stays_away_if_nobody_used_it(self):
        self.place.make_way()
        self.place.settled()
        self.assertFalse(self.card.running)
        self.assertIs(self.place.take(), self.card)  # but the next request may

    def test_a_load_that_never_reports_back_frees_the_card_after_the_hold(self):
        self.place.make_way()
        self.assertIs(self.place.take(), self.cpu)
        self.clock.now += HOLD_LIMIT_S + 1
        self.assertIs(self.place.take(), self.card)

    def test_by_hand_and_when_idle(self):
        self.place.start()
        self.assertTrue(self.card.running)
        self.free = 0.0
        self.place.stop()
        self.place.start()
        self.assertEqual((self.card.running, self.cpu.running), (False, True))
        self.clock.now += 1801
        self.assertEqual(self.place.stop_if_idle(busy=False), ["cpu"])
        self.assertFalse(self.place.running)
        self.assertEqual(self.place.device(), "stopped")


if __name__ == "__main__":
    unittest.main()
