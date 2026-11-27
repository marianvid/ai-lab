"""Every setting of the configuration file as a field of the Settings page."""

import json
import tempfile
import unittest
from dataclasses import fields as dataclass_fields
from pathlib import Path

from ai_lab.application.setting_fields import BY_KEY, FIELDS, GROUPS, SettingFields
from ai_lab.config.policy import GatewayPolicy, MachineLimits
from ai_lab.config.store import ConfigStore
from ai_lab.gpu_health.settings import GpuHealthSettings


PARSERS = {"limits": MachineLimits.from_mapping, "gateway": GatewayPolicy.from_mapping}


def parsed(key, value):
    """`value` at `key` alone, read by its section's own parser."""
    section, *path = key.split(".")
    raw: dict = {}
    holder = raw
    for part in path[:-1]:
        holder = holder.setdefault(part, {})
    holder[path[-1]] = value
    return PARSERS[section](raw)


class SettingFieldsTests(unittest.TestCase):
    def setUp(self):
        self._folder = tempfile.TemporaryDirectory()
        self.addCleanup(self._folder.cleanup)
        self.path = Path(self._folder.name) / "config.json"
        self.path.write_text(json.dumps({"limits": {"runtime": {"load_s": 1800}}}))
        self.told: list[str] = []
        self.fields = SettingFields(ConfigStore(self.path), self.told.append)

    def field(self, key):
        return next(item for item in self.fields.view()["fields"] if item["key"] == key)

    def test_every_value_of_the_limits_section_has_a_field_with_its_true_default(self):
        defaults = {"limits": MachineLimits(), "gateway": GatewayPolicy(),
                    "gpu_health": GpuHealthSettings()}
        for field in FIELDS:
            with self.subTest(field.key):
                section, *path = field.key.split(".")
                value = defaults[section]
                for part in path:
                    value = getattr(value, part)
                self.assertEqual(field.default, list(value) if isinstance(value, tuple) else value)
                self.assertIn(field.group, GROUPS)
        listed = {"runtime": 3, "builds": 6, "embeddings": 5, "clients": 2}
        for part, count in listed.items():
            self.assertEqual(sum(1 for key in BY_KEY if key.startswith(f"limits.{part}.")),
                             count, part)
        self.assertIn("limits.first_port", BY_KEY)
        self.assertIn("limits.editor_port_offset", BY_KEY)
        self.assertEqual(sum(1 for key in BY_KEY if key.startswith("gateway.max_upload_")), 3)
        # Every monitor setting but `channel`, which has one possible value today.
        monitor = {field.name for field in dataclass_fields(GpuHealthSettings)} - {"channel"}
        self.assertEqual({key.split(".", 1)[1] for key in BY_KEY if key.startswith("gpu_health.")},
                         monitor)

    def test_every_explanation_says_what_too_little_or_too_much_does(self):
        for field in FIELDS:
            with self.subTest(field.key):
                self.assertGreater(len(field.help), 60)

    def test_a_saved_value_is_shown_and_an_unsaved_one_shows_its_default(self):
        load, start = self.field("limits.runtime.load_s"), self.field("limits.runtime.start_s")
        self.assertEqual((load["value"], load["saved"]), (1800, True))
        self.assertEqual((start["value"], start["saved"]), (60.0, False))
        self.assertIn("next started", load["restart_note"])
        self.assertEqual(self.field("limits.first_port")["restart_note"], "")

    def test_a_change_is_saved_where_ai_lab_reads_it_and_the_page_is_told(self):
        self.fields.update({"limits.embeddings.call_s": 900, "limits.builds.system_path": " /opt/bin "})
        saved = json.loads(self.path.read_text())["limits"]
        self.assertEqual(saved["embeddings"], {"call_s": 900})
        self.assertEqual(saved["builds"]["system_path"], "/opt/bin")
        self.assertEqual(saved["runtime"], {"load_s": 1800})          # left as it was
        self.assertEqual(self.told, ["settings"])
        self.assertEqual(ConfigStore(self.path).load().machine_limits.embeddings.call_s, 900)

    def test_an_upload_cap_is_saved_in_the_gateway_section_and_checked_there(self):
        self.fields.update({"gateway.max_upload_pixels": 40_000_000})
        self.assertEqual(ConfigStore(self.path).load().gateway_policy.max_upload_pixels,
                         40_000_000)
        with self.assertRaisesRegex(ValueError, "Longest side must be at most 100000"):
            self.fields.update({"gateway.max_upload_dimension": 100001})

    def test_a_value_that_would_break_the_card_monitor_is_refused(self):
        before = self.path.read_text()
        breaking = {"gpu_health.sample_s": 0, "gpu_health.repeat_s": 0,
                    "gpu_health.critical_repeat_s": 59, "gpu_health.raw_keep_days": 0,
                    "gpu_health.r_alarm": 1.5}
        for key, value in breaking.items():
            with self.subTest(key), self.assertRaisesRegex(ValueError, BY_KEY[key].label):
                self.fields.update({key: value})
        self.assertEqual(self.path.read_text(), before)

    def test_a_pair_cannot_be_put_out_of_order_even_in_two_saves(self):
        self.fields.update({"gpu_health.stable_max_s": 900})
        with self.assertRaisesRegex(ValueError, "Stable window, shortest .1000. must be at most "
                                                "Stable window, longest .900."):
            self.fields.update({"gpu_health.stable_min_s": 1000})
        with self.assertRaisesRegex(ValueError, "R warning above reference .0.25. must be below"):
            self.fields.update({"gpu_health.r_warn": 0.25})
        with self.assertRaisesRegex(ValueError, "Idle window, shortest"):
            self.fields.update({"gpu_health.idle_max_s": 300})
        saved = ConfigStore(self.path).load().gpu_health
        self.assertEqual(saved, {"stable_max_s": 900})

    def test_the_page_bounds_of_limits_and_gateway_are_the_parsers_own(self):
        for field in FIELDS:
            if field.key.split(".")[0] not in PARSERS or field.kind not in ("number", "integer"):
                continue
            assert field.low is not None and field.high is not None
            step = 1 if field.kind == "integer" else 0.001
            with self.subTest(field.key):
                parsed(field.key, field.low)
                parsed(field.key, field.high)
                for outside in (field.low - step, field.high + step):
                    with self.assertRaises(ValueError):
                        parsed(field.key, outside)

    def test_the_card_monitor_takes_on_off_a_list_and_an_empty_path(self):
        self.fields.update({"gpu_health.enabled": False, "gpu_health.xid_critical": [48, 79],
                            "gpu_health.helper_events_path": "", "gpu_health.temp_gpu_c": 84})
        monitor = GpuHealthSettings.from_mapping(ConfigStore(self.path).load().gpu_health)
        self.assertEqual((monitor.enabled, monitor.xid_critical, monitor.helper_events_path,
                          monitor.temp_gpu_c), (False, (48, 79), "", 84.0))
        self.assertEqual(self.field("gpu_health.xid_critical")["value"], [48, 79])
        for changes, message in (({"gpu_health.enabled": 1}, "Monitor on must be on or off"),
                                 ({"gpu_health.xid_critical": [48, "x"]}, "whole numbers"),
                                 ({"gpu_health.sample_s": -1}, "Read every must be at least 1"),
                                 ({"gpu_health.pci_address": " "}, "must be text")):
            with self.subTest(message), self.assertRaisesRegex(ValueError, message):
                self.fields.update(changes)

    def test_a_wrong_value_is_refused_and_nothing_is_written(self):
        before = self.path.read_text()
        cases: dict[str, dict] = {
            "Unknown settings: limits.nonsense": {"limits.nonsense": 1},
            "Model load must be a number": {"limits.runtime.load_s": "long"},
            "Retry after must be a whole number": {"limits.clients.retry_after_s": 2.5},
            "Model load must be at least 1": {"limits.runtime.load_s": 0},
            "Limits builds system_path": {"limits.builds.system_path": "bin"},
            "Send the fields to change": {},
        }
        for message, changes in cases.items():
            with self.subTest(message), self.assertRaisesRegex(ValueError, message):
                self.fields.update(changes)
        self.assertEqual(self.path.read_text(), before)
        self.assertEqual(self.told, [])


if __name__ == "__main__":
    unittest.main()
