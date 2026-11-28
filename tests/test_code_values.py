"""The values still fixed in code, shown read-only (audit L19, L26, L34, L57, L59, L60, L61, m26)."""

import json
import unittest
from typing import cast

from ai_lab.application import code_values
from ai_lab.application.setting_fields import SettingFields
from ai_lab.builds import package_install
from ai_lab.config.store import ConfigStore
from tests.support import temporary


class CodeValuesTests(unittest.TestCase):
    def test_every_entry_still_names_a_constant_of_its_module(self):
        for item in code_values.VALUES:
            with self.subTest(item.key):
                self.assertIsInstance(code_values.value_of(item.module, item.name),
                                      (int, float, str, list))

    def test_the_value_shown_is_the_one_the_code_uses(self):
        self.assertEqual(code_values.value_of("builds/package_install.py", "CHECK_TIMEOUT_S"),
                         package_install.CHECK_TIMEOUT_S)
        self.assertEqual(code_values.value_of("gpu_health/monitor.py", "WEEK_S"), 7 * 86400.0)
        with self.assertRaises(KeyError):
            code_values.value_of("builds/package_install.py", "NOT_THERE")

    def test_a_language_and_a_table_are_shown_as_the_code_uses_them(self):
        # Audit L59 and L60: Canary's language and the models with their own editor.
        self.assertEqual(code_values.value_of("audio/server.py", "CANARY_LANGUAGE"), "ro")
        editors = cast(list, code_values.value_of("comfyui_templates/__init__.py", "_TEMPLATES"))
        self.assertIn("qwen-image-2512-nvfp4", editors)
        self.assertEqual(editors, sorted(editors))
        self.assertEqual(code_values.value_of("songs/studio.py", "STUDIO_JOBS_LOOKED_AT"), 50)

    def test_the_page_shows_them_and_refuses_to_change_them(self):
        path = temporary.folder(self) / "config.json"
        path.write_text(json.dumps({}))
        fields = SettingFields(ConfigStore(path), lambda _topic: None)
        shown = [item for item in fields.view()["fields"] if item["group"] == code_values.GROUP]
        self.assertEqual(len(shown), len(code_values.VALUES))
        self.assertTrue(all(item["kind"] == "fixed" for item in shown))
        with self.assertRaisesRegex(ValueError, "Fixed in AI-Lab's code"):
            fields.update({shown[0]["key"]: 1})


if __name__ == "__main__":
    unittest.main()
