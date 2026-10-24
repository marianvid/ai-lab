"""The anti-spam: once every 6 hours, critical always, "resolved" at the end."""

import threading
import unittest

from ai_lab.gpu_health.alarms import AlarmBook, Condition

from .fakes import settings, temporary_store

HOT = Condition("temp_gpu", "alarm", "Chip hot")
LOST = Condition("ecc_uncorrected", "critical", "Memory error")


class Book(unittest.TestCase):
    def setUp(self):
        self.store = temporary_store(self)
        self.sent: list[dict] = []
        self.book = AlarmBook(self.store, settings(), self.sent.append)

    def messages(self):
        return [(event["kind"], event["message"]) for event in self.sent]

    def test_a_condition_is_sent_once_then_every_six_hours_then_resolved(self):
        for now in (0, 15, 3600, 6 * 3600 + 1):
            self.book.conditions("fast", now, [HOT])
        self.book.conditions("fast", 6 * 3600 + 20, [])
        self.assertEqual(self.messages(), [("alarm", "Chip hot"), ("alarm", "Chip hot"),
                                           ("resolved", "Resolved: Chip hot")])
        self.assertEqual(self.book.active(), [])

    def test_an_alarm_back_after_resolved_is_sent_at_once(self):
        self.book.conditions("fast", 0, [HOT])
        self.book.conditions("fast", 100, [])
        self.book.conditions("fast", 700, [HOT])
        self.assertEqual(self.messages(), [("alarm", "Chip hot"), ("resolved", "Resolved: Chip hot"),
                                           ("alarm", "Chip hot")])

    def test_a_held_back_event_key_also_holds_back_its_condition(self):
        # The same key sent as an event a moment ago: the condition waits, and ends quietly.
        self.book.event(0, HOT)
        self.book.conditions("fast", 10, [HOT])
        self.book.conditions("fast", 20, [])
        self.assertEqual(len(self.sent), 1)
        kinds = [event["kind"] for event in self.store.events()]
        self.assertEqual(kinds, ["resolved", "alarm", "event"])

    def test_getting_worse_is_sent_at_once(self):
        self.book.conditions("trend", 0, [Condition("r_rise:global", "warning", "R +16 %")])
        self.book.conditions("trend", 3600, [Condition("r_rise:global", "alarm", "R +26 %")])
        self.book.conditions("trend", 3900, [Condition("r_rise:global", "critical", "R +40 %")])
        self.book.conditions("trend", 4000, [Condition("r_rise:global", "critical", "R +40 %")])
        self.assertEqual([event["level"] for event in self.sent], ["warning", "alarm", "critical"])

    def test_the_page_and_the_monitor_at_the_same_time(self):
        errors: list[BaseException] = []

        def monitor():
            for now in range(300):
                found = [Condition(f"k{index}", "warning", "x") for index in range(now % 7)]
                self.book.conditions("fast", float(now), found)

        def page():
            for now in range(300):
                try:
                    self.book.active()
                    self.book.event(float(now), Condition(f"test:{now}", "warning", "test"))
                except Exception as error:  # noqa: BLE001 - recorded for the test
                    errors.append(error)

        threads = [threading.Thread(target=monitor), threading.Thread(target=page)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(errors, [])

    def test_critical_conditions_repeat_every_fifteen_minutes(self):
        for now in (0, 600, 901, 1000, 1802):
            self.book.conditions("slow", now, [Condition("remap_failure", "critical", "x")])
        self.assertEqual(len(self.sent), 3)

    def test_critical_events_are_never_held_back(self):
        for now in (0, 1, 2):
            self.book.event(now, LOST)
        self.assertEqual(len(self.sent), 3)

    def test_ordinary_events_are_held_back_for_six_hours_but_written(self):
        warning = Condition("xid:13", "warning", "Xid 13")
        for now in (0, 10, 6 * 3600):
            self.book.event(now, warning)
        self.assertEqual(len(self.sent), 2)
        self.assertEqual(len(self.store.events()), 3)
        self.assertFalse(self.store.events()[1]["detail"]["sent"])

    def test_information_is_shown_never_sent(self):
        self.book.conditions("trend", 0, [Condition("idle_rise", "info", "warmer")])
        self.assertEqual(self.sent, [])
        self.assertEqual(self.book.active()[0]["key"], "idle_rise")

    def test_families_resolve_only_their_own(self):
        self.book.conditions("fast", 0, [HOT])
        self.book.conditions("slow", 1, [])
        self.assertEqual(len(self.book.active()), 1)

    def test_active_alarms_survive_a_restart_worst_first(self):
        self.book.conditions("fast", 0, [HOT])
        self.book.conditions("slow", 5, [Condition("remap_pending", "alarm", "pending"),
                                         Condition("remap_failure", "critical", "failed")])
        again = AlarmBook(self.store, settings(), self.sent.append)
        self.assertEqual([item["key"] for item in again.active()],
                         ["remap_failure", "temp_gpu", "remap_pending"])
        again.conditions("fast", 10, [HOT])
        self.assertEqual(len(self.sent), 3)  # not sent again: it was known
