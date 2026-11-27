"""The hosts' hand lists of engines name only engines the registry has (audit L18)."""

import unittest

from ai_lab.engines.registry import Registry
from ai_lab.hosts import darwin, linux


class EngineListsTests(unittest.TestCase):
    def test_every_engine_a_host_names_is_one_the_registry_knows(self):
        known = set(Registry().known())
        lists = {"LINUX_ENGINES": set(linux.LINUX_ENGINES),
                 "SUPPORTED_ENGINES": set(linux.SUPPORTED_ENGINES),
                 "DARWIN_ENGINES": set(darwin.DARWIN_ENGINES)}
        for name, engines in lists.items():
            with self.subTest(name):
                self.assertEqual(engines - known, set(), f"{name} names an unknown engine")
        self.assertLessEqual(set(linux.LINUX_ENGINES), set(linux.SUPPORTED_ENGINES))


if __name__ == "__main__":
    unittest.main()
