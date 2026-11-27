"""Versions in packaging's order (audit m25, L32)."""

import unittest

from ai_lab.changes.frompackages import Move
from ai_lab.changes.versions import version_key


class VersionKeyTests(unittest.TestCase):
    def test_packaging_order(self):
        ordered = ["0.27.0", "0.27.1.dev3", "0.27.1a1", "0.27.1rc2.dev949", "0.27.1rc2",
                   "0.27.1", "0.27.1.post1", "0.27.2", "0.28", "1.0"]
        keys = [version_key(text) for text in ordered]
        self.assertEqual(keys, sorted(keys))
        self.assertEqual(len(set(keys)), len(keys))

    def test_equal_spellings_and_what_is_not_a_version(self):
        self.assertEqual(version_key("v0.2.0"), version_key("0.2"))
        self.assertEqual(version_key("0.27.1+cu129"), version_key("0.27.1"))
        self.assertIsNone(version_key("nightly"))
        self.assertIsNone(version_key(""))

    def test_a_candidate_to_its_release_is_forwards(self):
        self.assertFalse(Move("vllm", "0.27.1rc2", "0.27.1").backwards)
        self.assertTrue(Move("torch", "2.10.0", "2.9.1").backwards)
        self.assertFalse(Move("new", "", "1.0").backwards)


if __name__ == "__main__":
    unittest.main()
