"""The SQLite file: what goes in comes out, and old samples fold into 5-minute rows."""

import unittest

from ai_lab.gpu_health.readings import Sample

from .fakes import temporary_store


class Store(unittest.TestCase):
    def setUp(self):
        self.store = temporary_store(self)

    def test_samples_in_order_and_the_last(self):
        for ts in (30.0, 10.0, 20.0):
            self.store.add_sample(Sample(ts=ts, temp_gpu=ts, model="m", running=True))
        self.assertEqual([row["ts"] for row in self.store.samples_since(15)], [20.0, 30.0])
        last = self.store.last_sample()
        assert last is not None
        self.assertEqual(last["running"], 1)
        self.assertEqual(self.store.count("samples"), 3)

    def test_folding_keeps_min_max_mean_and_deletes_the_raw(self):
        for index in range(40):  # 10 minutes of readings, from ts 0
            self.store.add_sample(Sample(ts=index * 15.0, temp_gpu=30 + index % 4,
                                         power_w=100.0, fan_pct=40.0, util_pct=50.0))
        made = self.store.fold_old_samples(610.0)  # moved back to 600: two whole buckets
        self.assertEqual(made, 2)
        rows = self.store.samples_5m_since(0)
        self.assertEqual([row["ts"] for row in rows], [0.0, 300.0])
        self.assertEqual((rows[0]["n"], rows[0]["temp_min"], rows[0]["temp_max"]), (20, 30, 33))
        self.assertEqual(self.store.count("samples"), 0)

    def test_windows_by_day_and_model(self):
        for day, model in (("2026-10-01", "a"), ("2026-10-02", "b")):
            self.store.add_window({"start": 1, "end": 2, "day": day, "model": model,
                                   "temp_avg": 60, "power_avg": 150})
        self.assertEqual(len(self.store.windows("2026-10-02")), 1)
        self.assertEqual(self.store.windows("", "a")[0]["day"], "2026-10-01")
        self.store.add_idle_window(1, 2, "2026-10-01", 30.0)
        self.assertEqual(self.store.idle_windows("")[0]["temp_median"], 30.0)

    def test_events_newest_first_with_their_detail(self):
        self.store.add_event({"ts": 1, "kind": "event", "message": "a", "detail": {"x": 1}})
        self.store.add_event({"ts": 2, "kind": "event", "message": "b"})
        events = self.store.events()
        self.assertEqual([event["message"] for event in events], ["b", "a"])
        self.assertEqual(events[1]["detail"], {"x": 1})
        self.assertIsNone(events[0]["detail"])

    def test_baselines_slow_readings_and_state(self):
        self.store.save_baseline({"scope": "global", "r": 0.2, "fan_bands": {"150": 40},
                                  "windows": 12, "period_start": "a", "period_end": "b",
                                  "formed_at": 1})
        self.assertEqual(self.store.baselines()["global"]["fan_bands"], {"150": 40})
        self.store.clear_baselines()
        self.assertEqual(self.store.baselines(), {})
        self.store.add_slow_reading(5, {"corrected_aggregate": 1}, None, "610")
        self.assertEqual(self.store.slow_readings(0)[0]["ecc"], {"corrected_aggregate": 1})
        self.assertIsNone(self.store.slow_readings(0)[0]["rows"])
        self.assertEqual(self.store.state("missing", 7), 7)
        self.store.set_state("k", {"a": [1]})
        self.assertEqual(self.store.state("k"), {"a": [1]})

    def test_an_unknown_table_is_refused(self):
        with self.assertRaises(ValueError):
            self.store.count("samples; DROP TABLE samples")
