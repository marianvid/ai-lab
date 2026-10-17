"""Who holds the card, who waits, and when a reservation ends (ADR 0005)."""

import tempfile
import unittest
from pathlib import Path

from ai_lab.reservations.book import ReservationBook
from ai_lab.reservations.errors import (
    CardReservedError,
    ReservationEndedError,
    ReservationNotGrantedError,
    ReservationRequiredError,
)
from ai_lab.reservations.settings import ReservationPolicy


class Clock:
    def __init__(self):
        self.now = 1_000_000.0

    def __call__(self):
        return self.now


POLICY = ReservationPolicy(unreserved_wait_s=0)


class BookTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.path = Path(tempfile.mkdtemp()) / "reservations.json"
        self.book = ReservationBook(POLICY, self.path, self.clock)

    def ask(self, team="ingest", priority=3, **more):
        return self.book.request({"team": team, "priority": priority, **more})

    def held(self) -> dict:
        holder = self.book.holder()
        assert holder is not None
        return holder

    def later(self, seconds):
        self.clock.now += seconds

    def test_the_first_is_granted_the_next_waits(self):
        first = self.ask(instance="ct109", purpose="collect", run_id="r1")
        second = self.ask("music")
        self.assertEqual((first["state"], first["time_left_s"]), ("granted", 1800))
        self.assertEqual((second["state"], second["position"]), ("queued", 1))
        self.assertEqual(self.held()["run_id"], "r1")

    def test_higher_priority_goes_next_same_priority_in_arrival_order(self):
        self.ask("holder")
        low = self.ask("low", 2)
        self.later(1)
        early = self.ask("early", 4)
        self.later(1)
        late = self.ask("late", 4)
        queue = [item["team"] for item in self.book.overview()["queue"]]
        self.assertEqual(queue, ["early", "late", "low"])
        self.book.release(self.held()["id"])
        self.assertEqual(self.held()["id"], early["id"])
        self.assertEqual(self.book.get(late["id"])["position"], 1)
        self.assertEqual(self.book.get(low["id"])["position"], 2)

    def test_waiting_raises_priority_so_nothing_waits_forever(self):
        holder = self.ask("holder")
        self.book.admit(holder["id"])
        old = self.ask("old", 1)
        for _ in range(3):              # keep both alive while time passes
            self.later(150)
            self.book.heartbeat(holder["id"])
            self.book.heartbeat(old["id"])
        self.later(150)
        self.book.heartbeat(holder["id"])
        self.book.heartbeat(old["id"])   # 600 s waited: one level up
        new = self.ask("new", 2)
        self.assertEqual(self.book.get(old["id"])["effective_priority"], 2)
        self.assertEqual([item["id"] for item in self.book.overview()["queue"]],
                         [old["id"], new["id"]])

    def test_no_heartbeat_frees_the_card(self):
        holder = self.ask()
        waiting = self.ask("music")
        self.book.heartbeat(waiting["id"])
        self.later(100)
        self.book.heartbeat(waiting["id"])
        self.later(100)                  # holder silent for 200 s > 180 s
        self.book.heartbeat(waiting["id"])
        ended = self.book.get(holder["id"])
        self.assertEqual(ended["state"], "ended")
        self.assertIn("no heartbeat", ended["reason"])
        self.assertEqual(self.held()["id"], waiting["id"])

    def test_unused_counts_only_until_the_first_call(self):
        idle = self.ask()
        for _ in range(3):
            self.later(170)
            self.book.heartbeat(idle["id"])
        self.later(100)                  # 610 s granted, never called
        self.assertIn("unused", self.book.get(idle["id"])["reason"])
        used = self.ask()
        self.book.admit(used["id"])
        for _ in range(8):               # a long gap between steps, still beating
            self.later(170)
            self.book.heartbeat(used["id"])
        self.assertEqual(self.book.get(used["id"])["state"], "granted")

    def test_maximum_duration_and_extending_up_to_the_ceiling(self):
        held = self.ask(max_s=300)
        self.book.admit(held["id"])
        extended = self.book.extend(held["id"], 10**6)
        self.assertEqual(extended["max_s"], POLICY.ceiling_s)
        short = ReservationBook(POLICY, None, self.clock)
        one = short.request({"team": "a", "max_s": 120})
        short.admit(one["id"])
        self.later(121)
        self.assertEqual(short.get(one["id"])["reason"], "reached its maximum duration")

    def test_calls_under_other_reservations_are_refused_with_the_reason(self):
        holder = self.ask()
        waiting = self.ask("music")
        self.book.admit(holder["id"])
        with self.assertRaisesRegex(ReservationNotGrantedError, "position 1"):
            self.book.admit(waiting["id"])
        self.book.release(holder["id"])
        with self.assertRaisesRegex(ReservationEndedError, "released by its holder"):
            self.book.admit(holder["id"])
        with self.assertRaisesRegex(ReservationEndedError, "unknown"):
            self.book.admit("nope")
        with self.assertRaises(ReservationEndedError):
            self.book.heartbeat(holder["id"])

    def test_a_call_without_a_reservation(self):
        self.book.admit(None)            # nothing reserved: goes ahead
        self.ask()
        with self.assertRaisesRegex(CardReservedError, "reserved by ingest"):
            self.book.admit(None)
        strict = ReservationBook(ReservationPolicy(required=True), None, self.clock)
        with self.assertRaises(ReservationRequiredError):
            strict.admit(None)

    def test_cancel_and_history(self):
        held = self.ask()
        cancelled = self.book.cancel(held["id"])
        self.assertEqual(cancelled["reason"], "cancelled from the AI-Lab page")
        overview = self.book.overview()
        self.assertIsNone(overview["holder"])
        self.assertEqual(overview["history"][0]["id"], held["id"])
        self.assertEqual(overview["settings"]["ceiling_s"], 14400)
        with self.assertRaises(KeyError):
            self.book.cancel("nope")

    def test_a_restart_restores_the_holder(self):
        held = self.ask()
        self.later(500)                  # the manager was down for a while
        again = ReservationBook(POLICY, self.path, self.clock)
        restored = again.get(held["id"])
        self.assertEqual((restored["state"], restored["restored"]), ("granted", True))
        self.path.write_text("not json")
        self.assertIsNone(ReservationBook(POLICY, self.path, self.clock).holder())

    def test_bad_requests_are_refused_by_name(self):
        for body, words in (({}, "team is required"),
                            ({"team": "a", "priority": 9}, "priority"),
                            ({"team": "a", "max_s": 10**9}, "max_s"),
                            ({"team": 5}, "team")):
            with self.subTest(body=body), self.assertRaisesRegex(ValueError, words):
                self.book.request(body)
        held = self.ask()
        with self.assertRaisesRegex(ValueError, "seconds"):
            self.book.extend(held["id"], 0)

    def test_new_settings_apply_at_once(self):
        held = self.ask()
        self.book.apply_policy(ReservationPolicy(heartbeat_timeout_s=30))
        self.later(31)
        self.book.settle()
        self.assertEqual(self.book.get(held["id"])["state"], "ended")


if __name__ == "__main__":
    unittest.main()
