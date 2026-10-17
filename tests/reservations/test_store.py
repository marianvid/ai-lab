"""A reservation saved to disk and read back, and a damaged file set aside."""

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from ai_lab.reservations.reservation import Reservation, checked_priority, checked_text
from ai_lab.reservations.store import ReservationFile


def sample(**changes) -> Reservation:
    first = Reservation(id="r1", team="ingest", priority=4, max_s=1800,
                        requested_at=10.0, last_beat=10.0)
    return replace(first, **changes)


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.path = Path(tempfile.mkdtemp()) / "state" / "reservations.json"
        self.file = ReservationFile(self.path)

    def test_nothing_saved_reads_as_empty(self):
        self.assertEqual(self.file.load(), [])
        self.assertEqual(ReservationFile(None).load(), [])

    def test_saved_reservations_come_back_equal(self):
        saved = [sample(), sample(id="r2", team="music", priority=2)]
        self.file.save(saved)
        self.assertEqual(self.file.load(), saved)
        self.assertFalse(self.path.with_suffix(".tmp").exists())

    def test_an_unreadable_file_is_set_aside(self):
        self.path.parent.mkdir(parents=True)
        self.path.write_text("{not json")
        self.assertEqual(self.file.load(), [])
        self.assertTrue(self.path.with_suffix(".unreadable").is_file())

    def test_unknown_keys_on_disk_are_ignored(self):
        record = {**sample().record(), "from_a_newer_version": True}
        self.assertEqual(Reservation.from_record(record), sample())


class CheckTests(unittest.TestCase):
    def test_priority_is_a_whole_number_from_1_to_5(self):
        self.assertEqual(checked_priority(5), 5)
        for wrong in (0, 6, 2.5, True, "3"):
            with self.assertRaises(ValueError):
                checked_priority(wrong)

    def test_labels_are_short_text_and_required_ones_not_blank(self):
        self.assertEqual(checked_text({"team": " ingest "}, "team", required=True), "ingest")
        self.assertEqual(checked_text({}, "purpose"), "")
        for body in ({"team": "  "}, {"team": 7}, {"team": "x" * 201}):
            with self.assertRaises(ValueError):
                checked_text(body, "team", required=True)
