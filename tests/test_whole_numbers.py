"""A whole number is never a bool or a fraction cut down (audit L35, L52)."""

import unittest

from ai_lab.clients.security import SecuritySettings
from ai_lab.gpu_health.settings import GpuHealthSettings
from ai_lab.shared.numbers import whole_number
from tests.support import temporary


class WholeNumberTests(unittest.TestCase):
    def test_what_is_and_is_not_a_whole_number(self):
        self.assertEqual([whole_number(v, "x") for v in (3, 3.0, "12", -2)], [3, 3, 12, -2])
        for wrong in (True, 3.9, "3.9", "many", None):
            with self.subTest(wrong=wrong), self.assertRaises(ValueError):
                whole_number(wrong, "x")

    def test_the_security_numbers_refuse_a_fraction_and_a_bool(self):
        security = SecuritySettings(temporary.folder(self) / "security.json")
        for wrong in (3.9, True):
            with self.subTest(wrong=wrong), self.assertRaises(ValueError):
                security.update({"session_days": wrong})
        security.update({"session_days": 7.0})
        self.assertEqual(security.view()["session_days"], 7)

    def test_a_gpu_health_count_refuses_a_fraction(self):
        with self.assertRaisesRegex(ValueError, "whole number"):
            GpuHealthSettings.from_mapping({"baseline_min_windows": 10.5})


if __name__ == "__main__":
    unittest.main()
