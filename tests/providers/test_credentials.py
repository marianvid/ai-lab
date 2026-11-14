"""The vendor's private environment file: read for the CLI, described for the page."""

import os
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from ai_lab.providers import credentials

SECRET = "stand-in-for-a-yearly-token"


class CredentialFileTests(unittest.TestCase):
    def setUp(self):
        self._temporary = TemporaryDirectory()
        self.addCleanup(self._temporary.cleanup)
        self.path = Path(self._temporary.name) / "claude-token.env"

    def write(self, text, mode=0o600, age_days=0.0):
        self.path.write_text(text)
        self.path.chmod(mode)
        when = time.time() - age_days * 86400
        os.utime(self.path, (when, when))
        return str(self.path)

    def test_no_file_named_or_no_file_there_means_the_login(self):
        self.assertEqual(credentials.read(""), {})
        self.assertEqual(credentials.describe(""), {"credential": "login"})
        self.assertEqual(credentials.read(str(self.path)), {})
        self.assertEqual(credentials.describe(str(self.path)),
                         {"credential": "login", "env_file": str(self.path)})

    def test_an_empty_file_means_the_login(self):
        path = self.write("# put the token here\n\n")
        self.assertEqual(credentials.read(path), {})
        self.assertEqual(credentials.describe(path)["credential"], "login")

    def test_the_token_line_is_read_in_its_usual_spellings(self):
        for text in (f"CLAUDE_CODE_OAUTH_TOKEN={SECRET}\n",
                     f"export CLAUDE_CODE_OAUTH_TOKEN={SECRET}",
                     f'CLAUDE_CODE_OAUTH_TOKEN="{SECRET}"\n',
                     f"  CLAUDE_CODE_OAUTH_TOKEN = '{SECRET}'  \n"):
            with self.subTest(text=text):
                path = self.write(text)
                self.assertEqual(credentials.read(path), {"CLAUDE_CODE_OAUTH_TOKEN": SECRET})

    def test_the_description_names_the_token_and_its_date_but_never_its_value(self):
        path = self.write(f"CLAUDE_CODE_OAUTH_TOKEN={SECRET}\n", age_days=10.2)
        state = credentials.describe(path)
        self.assertEqual(state["credential"], "token")
        self.assertEqual(state["age_days"], 10)
        self.assertRegex(state["set_on"], r"^\d{4}-\d{2}-\d{2}$")
        self.assertNotIn("warning", state)
        self.assertNotIn(SECRET, repr(state))

    def test_a_warning_from_eleven_months_on(self):
        path = self.write(f"CLAUDE_CODE_OAUTH_TOKEN={SECRET}\n",
                          age_days=credentials.TOKEN_WARNING_AFTER_DAYS + 1)
        self.assertIn("claude setup-token", credentials.describe(path)["warning"])
        path = self.write(f"CLAUDE_CODE_OAUTH_TOKEN={SECRET}\n", age_days=400)
        self.assertIn("may have ended", credentials.describe(path)["warning"])

    def test_a_file_others_can_read_is_still_used_but_flagged(self):
        path = self.write(f"CLAUDE_CODE_OAUTH_TOKEN={SECRET}\n", mode=0o644)
        self.assertEqual(credentials.read(path), {"CLAUDE_CODE_OAUTH_TOKEN": SECRET})
        self.assertIn("chmod 600", credentials.describe(path)["warning"])

    def test_a_line_not_understood_is_named_by_number_not_shown(self):
        path = self.write(f"CLAUDE_CODE_OAUTH_TOKEN={SECRET}\n{SECRET}\n")
        state = credentials.describe(path)
        self.assertIn("line(s) 2", state["warning"])
        self.assertNotIn(SECRET, repr(state))
        self.assertEqual(credentials.read(path), {"CLAUDE_CODE_OAUTH_TOKEN": SECRET})

    def test_a_file_that_cannot_be_read_falls_back_to_the_login(self):
        path = self.write(f"CLAUDE_CODE_OAUTH_TOKEN={SECRET}\n")
        if os.geteuid() == 0:
            self.skipTest("root reads any file")
        self.path.chmod(0)
        self.addCleanup(self.path.chmod, 0o600)
        self.assertEqual(credentials.read(path), {})
        state = credentials.describe(path)
        self.assertEqual(state["credential"], "login")
        self.assertIn("cannot be read", state["warning"])


if __name__ == "__main__":
    unittest.main()
