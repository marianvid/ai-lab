"""The weekly report: the numbers, and the same as sentences."""

import unittest

from ai_lab.gpu_health.analysis.references import Analysis
from ai_lab.gpu_health.readings import Sample
from ai_lab.gpu_health.output.report import text, weekly

from .fakes import settings, temporary_store

NOW = 1_800_000_000.0


class Report(unittest.TestCase):
    def setUp(self):
        self.store = temporary_store(self)
        self.analysis = Analysis(self.store, settings())

    def test_an_empty_week(self):
        report = weekly(self.store, self.analysis, NOW)
        self.assertEqual((report["samples"], report["temp"], report["alarms"]), (0, None, 0))
        self.assertIn("No thermal resistance", report["text"])
        self.assertIn("no reference.", report["text"])
        self.assertIn("No alarms.", report["text"])

    def test_a_week_with_readings_and_alarms(self):
        for index in range(10):
            self.store.add_sample(Sample(ts=NOW - 1000 + index, temp_gpu=40 + index,
                                         power_w=100.0 + index))
        self.store.add_event({"ts": NOW - 10, "kind": "event", "level": "critical",
                              "message": "Xid 79"})
        self.store.add_event({"ts": NOW - 10, "kind": "baseline", "level": "info",
                              "message": "formed"})
        report = weekly(self.store, self.analysis, NOW)
        self.assertEqual(report["temp"], {"min": 40, "max": 49, "mean": 44.5})
        self.assertEqual((report["alarms"], report["critical"]), (1, 1))
        self.assertIn("Chip 40–49 °C", report["text"])
        self.assertIn("Alarms: 1 (1 critical).", report["text"])

    def test_a_report_longer_than_the_raw_readings_counts_the_folded_ones_too(self):
        # Audit L38: only the raw part was counted once any raw row existed.
        for index in range(12):
            self.store.add_sample(Sample(ts=NOW - 6 * 86400 + index * 30, temp_gpu=50, power_w=100.0))
        self.store.fold_old_samples(NOW - 5 * 86400)
        for index in range(5):
            self.store.add_sample(Sample(ts=NOW - 100 + index, temp_gpu=60, power_w=100.0))
        report = weekly(self.store, self.analysis, NOW)
        self.assertGreater(report["samples"], 5)
        self.assertEqual(report["temp"]["min"], 50)

    def test_r_against_its_reference_or_without_one(self):
        base = {"days": 7, "samples": 1, "windows": 3, "idle_windows": 2, "temp": None,
                "alarms": 0, "critical": 0, "idle_today": {"temp": 30.0, "from_day": "d"},
                "trend": {"global": {"r": 0.23}}}
        with_reference = text({**base, "baselines": {"global": {
            "r": 0.2, "period_start": "a", "period_end": "b"}}})
        self.assertIn("+15% against the reference 0.200", with_reference)
        self.assertIn("Resting temperature today: 30 °C", with_reference)
        self.assertIn("no reference yet", text({**base, "baselines": {}}))
