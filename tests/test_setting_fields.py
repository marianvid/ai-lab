"""Every setting of the configuration file as a field of the Settings page."""

import json
import tempfile
import unittest
from pathlib import Path

from ai_lab.application.setting_fields import BY_KEY, FIELDS, GROUPS, SettingFields
from ai_lab.config.policy import MachineLimits
from ai_lab.config.store import ConfigStore


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
        defaults = MachineLimits()
        for field in FIELDS:
            with self.subTest(field.key):
                value = defaults
                for part in field.key.split(".")[1:]:
                    value = getattr(value, part)
                self.assertEqual(field.default, value)
                self.assertIn(field.group, GROUPS)
        listed = {"runtime": 3, "builds": 6, "embeddings": 5, "clients": 2}
        for part, count in listed.items():
            self.assertEqual(sum(1 for key in BY_KEY if key.startswith(f"limits.{part}.")),
                             count, part)
        self.assertIn("limits.first_port", BY_KEY)

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

    def test_a_wrong_value_is_refused_and_nothing_is_written(self):
        before = self.path.read_text()
        cases: dict[str, dict] = {
            "Unknown settings: limits.nonsense": {"limits.nonsense": 1},
            "Model load must be a number": {"limits.runtime.load_s": "long"},
            "Retry after must be a whole number": {"limits.clients.retry_after_s": 2.5},
            "Limits runtime load_s must be between": {"limits.runtime.load_s": 0},
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
