import unittest

from ai_lab.engines.base import ParamSpec, validate


class ParamSpecTests(unittest.TestCase):
    def test_int_is_range_checked(self):
        spec = ParamSpec("ctx", "Context size", "int", 4096, minimum=512, maximum=8192)
        self.assertEqual(spec.coerce("2048"), 2048)
        with self.assertRaises(ValueError) as caught:
            spec.coerce(100)
        self.assertIn("at least 512", str(caught.exception))
        with self.assertRaises(ValueError):
            spec.coerce(99999)

    def test_int_rejects_nonsense(self):
        spec = ParamSpec("ctx", "Context size", "int", 4096)
        with self.assertRaises(ValueError) as caught:
            spec.coerce("plenty")
        self.assertIn("whole number", str(caught.exception))

    def test_choice_is_restricted(self):
        spec = ParamSpec("k", "Key cache", "choice", "q4_0", choices=("f16", "q4_0"))
        self.assertEqual(spec.coerce("f16"), "f16")
        with self.assertRaises(ValueError):
            spec.coerce("q9_9")

    def test_bool_requires_a_real_boolean(self):
        spec = ParamSpec("fa", "Flash attention", "bool", True)
        self.assertTrue(spec.coerce(True))
        with self.assertRaises(ValueError):
            spec.coerce("yes")


class ValidateTests(unittest.TestCase):
    specs = (
        ParamSpec("ctx", "Context size", "int", 4096, minimum=512),
        ParamSpec("fa", "Flash attention", "bool", True),
    )

    def test_missing_values_fall_back_to_defaults(self):
        self.assertEqual(validate(self.specs, {}), {"ctx": 4096, "fa": True})

    def test_supplied_values_win(self):
        self.assertEqual(validate(self.specs, {"ctx": 8192}), {"ctx": 8192, "fa": True})

    def test_unknown_settings_are_rejected(self):
        with self.assertRaises(ValueError) as caught:
            validate(self.specs, {"turbo": True})
        self.assertIn("turbo", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
