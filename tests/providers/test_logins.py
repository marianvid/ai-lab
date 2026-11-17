"""The last known state of a vendor's login, learned only from real calls."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from ai_lab.providers.logins import FILE_NAME, LoginBook
from ai_lab.providers.settings import ProvidersConfig


class LoginBookTests(unittest.TestCase):
    def setUp(self):
        self._temporary = TemporaryDirectory()
        self.addCleanup(self._temporary.cleanup)
        self.root = Path(self._temporary.name)
        self.times = iter(f"2026-11-17T14:0{n}:00Z" for n in range(10))

    def book(self):
        return LoginBook(self.root, clock=lambda: next(self.times))

    def test_unknown_before_any_call(self):
        self.assertEqual(self.book().state("anthropic"), {"state": "unknown"})

    def test_since_is_when_it_became_so_and_last_learned_moves_on(self):
        book = self.book()
        book.answered("anthropic", "haiku")
        book.answered("anthropic", "opus")
        state = book.state("anthropic")
        self.assertEqual((state["state"], state["since"], state["last_learned"], state["model"]),
                         ("ok", "2026-11-17T14:00:00Z", "2026-11-17T14:01:00Z", "opus"))
        book.refused("anthropic", "opus", "Not logged in")
        state = book.state("anthropic")
        self.assertEqual((state["state"], state["since"], state["said"]),
                         ("expired", "2026-11-17T14:02:00Z", "Not logged in"))
        book.answered("anthropic", "haiku")
        self.assertNotIn("said", book.state("anthropic"))

    def test_kept_in_a_file_only_when_the_state_changes(self):
        book = self.book()
        book.refused("openai", "gpt-5.6-terra", "Missing bearer")
        kept = json.loads((self.root / FILE_NAME).read_text())
        self.assertEqual(kept["openai"]["state"], "expired")
        self.assertEqual(LoginBook(self.root).state("openai")["state"], "expired")

    def test_a_damaged_file_is_read_as_nothing_known(self):
        (self.root / FILE_NAME).write_text("not json")
        self.assertEqual(LoginBook(self.root).state("anthropic"), {"state": "unknown"})
        (self.root / FILE_NAME).write_text("[1, 2]")
        self.assertEqual(LoginBook(self.root).state("anthropic"), {"state": "unknown"})


class MachineNameTests(unittest.TestCase):
    def test_the_machine_is_named_in_the_configuration(self):
        self.assertEqual(ProvidersConfig.from_mapping({"machine": " ai-lab-mac "}).machine,
                         "ai-lab-mac")
        self.assertEqual(ProvidersConfig.from_mapping({}).machine, "")
        with self.assertRaisesRegex(ValueError, "providers.machine"):
            ProvidersConfig.from_mapping({"machine": 7})


if __name__ == "__main__":
    unittest.main()
