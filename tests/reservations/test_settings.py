"""The `reservations` section: defaults, limits, and refusals by name."""

import unittest

from ai_lab.config.store import Config
from ai_lab.reservations.settings import ReservationPolicy


class PolicyTests(unittest.TestCase):
    def test_defaults(self):
        policy = ReservationPolicy.from_mapping(None)
        self.assertEqual((policy.heartbeat_timeout_s, policy.default_max_s, policy.ceiling_s,
                          policy.required), (180, 1800, 14400, False))
        self.assertEqual(Config(reservations={"aging_s": 0}).reservation_policy.aging_s, 0)

    def test_mistakes_are_refused(self):
        for raw, words in (("x", "must be an object"),
                           ({"nope": 1}, "Unknown"),
                           ({"unused_s": 5}, "unused_s"),
                           ({"required": "yes"}, "true or false"),
                           ({"default_max_s": 20000}, "may not exceed")):
            with self.subTest(raw=raw), self.assertRaisesRegex(ValueError, words):
                ReservationPolicy.from_mapping(raw)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
