"""Calls without a reservation waiting at the door are counted, not invisible."""

import threading
import time
import unittest

from ai_lab.reservations.book import ReservationBook
from ai_lab.reservations.door import DoorWaiters
from ai_lab.reservations.errors import CardReservedError
from ai_lab.reservations.settings import ReservationPolicy


class DoorWaitersTests(unittest.TestCase):
    def test_nobody_waiting(self):
        self.assertEqual(DoorWaiters().view(), {"calls": 0, "models": {}, "oldest_since": None})

    def test_calls_are_counted_per_model_with_the_oldest(self):
        times = iter([100.0, 105.0, 110.0])
        door = DoorWaiters(clock=lambda: next(times))
        first = door.arrive("qwen38-nvfp4")
        door.arrive("qwen38-nvfp4")
        door.arrive("")
        self.assertEqual(door.view(), {"calls": 3,
                                       "models": {"qwen38-nvfp4": 2, "": 1},
                                       "oldest_since": 100.0})
        door.leave(first)
        door.leave(first)  # a second leave changes nothing
        self.assertEqual(door.view()["oldest_since"], 105.0)
        self.assertEqual(door.view()["calls"], 2)


class BookDoorTests(unittest.TestCase):
    def wait_for(self, book, calls, seconds=3.0):
        deadline = time.time() + seconds
        while time.time() < deadline:
            if book.overview()["at_the_door"]["calls"] == calls:
                return True
            time.sleep(0.02)
        return False

    def test_a_call_waiting_on_a_held_card_shows_and_then_goes_in(self):
        book = ReservationBook(ReservationPolicy(unreserved_wait_s=5))
        holder = book.request({"team": "ingest", "priority": 4})
        done = threading.Event()

        def call():
            book.admit(None, model="qwen38-nvfp4")
            done.set()

        threading.Thread(target=call, daemon=True).start()
        self.assertTrue(self.wait_for(book, 1), "the waiting call was not counted")
        self.assertEqual(book.overview()["at_the_door"]["models"], {"qwen38-nvfp4": 1})
        book.release(holder["id"])
        self.assertTrue(done.wait(3), "the call did not go in once the card was free")
        self.assertTrue(self.wait_for(book, 0), "the call stayed counted after going in")

    def test_a_refused_call_is_no_longer_counted(self):
        book = ReservationBook(ReservationPolicy(unreserved_wait_s=0))
        book.request({"team": "ingest", "priority": 4})
        with self.assertRaises(CardReservedError):
            book.admit(None, model="qwen38-nvfp4")
        self.assertEqual(book.overview()["at_the_door"]["calls"], 0)

    def test_a_free_card_counts_nobody(self):
        book = ReservationBook(ReservationPolicy(unreserved_wait_s=5))
        book.admit(None, model="qwen38-nvfp4")
        self.assertEqual(book.overview()["at_the_door"]["calls"], 0)


class GatewayStatsTests(unittest.TestCase):
    def test_the_gateway_statistics_carry_the_door(self):
        from typing import Any, cast

        from ai_lab.api.routes.gateway import NO_LANES, _stats

        class Gateway:
            @staticmethod
            def stats():
                return {"waiting": 0}

        book = ReservationBook(ReservationPolicy())
        stats = _stats(cast(Any, Gateway()), NO_LANES, book)
        self.assertEqual(stats["at_the_door"]["calls"], 0)
        self.assertNotIn("at_the_door", _stats(cast(Any, Gateway()), NO_LANES))
