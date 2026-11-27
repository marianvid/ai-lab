"""Settings kept outside config.json are on a page too (the owner's rule; audit m22).

Security's switches and numbers live in security.json; the Access page must
offer each one. Read from the page's own source, so a new Security setting
without a field fails here.
"""

import tempfile
import unittest
from pathlib import Path

from ai_lab.clients.security import NUMBERS, SWITCHES, SecuritySettings

ACCESS_PAGE = Path(__file__).resolve().parent.parent / "ai_lab" / "web" / "js" / "views" / "access.js"
SHOWN_NOT_SET = {"password_set", "token_holds", "password_hash"}


class OutsideTheFileTests(unittest.TestCase):
    def test_every_security_setting_has_its_field_on_the_access_page(self):
        page = ACCESS_PAGE.read_text()
        for key in SWITCHES:
            with self.subTest(key):
                self.assertIn(f"toggle('{key}'", page)
        for key in NUMBERS:
            with self.subTest(key):
                self.assertIn(f"key: '{key}'", page)

    def test_the_view_holds_nothing_the_page_does_not_know(self):
        view = SecuritySettings(Path(tempfile.mkdtemp()) / "security.json").view()
        self.assertEqual(set(view) - set(SWITCHES) - set(NUMBERS), SHOWN_NOT_SET)


if __name__ == "__main__":
    unittest.main()
