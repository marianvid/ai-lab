"""A vendor's usage-limit state, as the console's band reads it."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from ai_lab.providers.limit_book import LimitBook, resets_at


class ResetsAtTests(unittest.TestCase):
    def test_claude_codes_epoch_after_a_bar(self):
        self.assertEqual(resets_at("Claude AI usage limit reached|1764158400"),
                         "2025-11-26T12:00:00Z")

    def test_an_iso_time_in_utc_or_with_an_offset(self):
        self.assertEqual(resets_at("try again at 2026-11-26T15:00:00Z"), "2026-11-26T15:00:00Z")
        self.assertEqual(resets_at("resets at 2026-11-26 17:00+02:00"), "2026-11-26T15:00:00Z")

    def test_words_without_a_time_give_nothing(self):
        self.assertEqual(resets_at("You've hit your usage limit. Try again later."), "")
        self.assertEqual(resets_at(""), "")


class LimitBookTests(unittest.TestCase):
    def setUp(self):
        self._temporary = TemporaryDirectory()
        self.addCleanup(self._temporary.cleanup)
        self.now = [1_800_000_000.0]
        self.book = LimitBook(Path(self._temporary.name), clock=lambda: self.now[0])

    def test_unknown_before_any_call(self):
        self.assertEqual(self.book.state("openai"), {"state": "unknown"})

    def test_limited_keeps_its_since_and_says_when_ai_lab_tries_again(self):
        first = self.book.limited("openai", "gpt", "usage", "usage limit", 120)
        self.now[0] += 60
        second = self.book.limited("openai", "gpt", "usage", "usage limit", 300)
        self.assertEqual(second["since"], first["since"])
        self.assertEqual(second["retry_at"], "2027-01-15T08:06:00Z")
        self.assertEqual(second["state"], "limited")

    def test_an_answer_ends_it_and_a_restart_remembers(self):
        self.book.limited("openai", "gpt", "rate", "429", 30)
        self.assertEqual(LimitBook(Path(self._temporary.name)).state("openai")["state"],
                         "limited")
        self.book.answered("openai", "gpt")
        state = LimitBook(Path(self._temporary.name)).state("openai")
        self.assertEqual((state["state"], state["said"], state["retry_at"]), ("ok", "", ""))
