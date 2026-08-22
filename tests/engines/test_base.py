import unittest

from ai_lab.engines.base import ANTHROPIC_PATHS, OPENAI_PATHS

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


class AdvertisedPathsTests(unittest.TestCase):
    """Only what works with the engine started as it is."""

    def test_embeddings_is_not_offered(self):
        """It was, and every model refused it.

        llama.cpp answers `/v1/embeddings` only when started with
        `--embeddings`, which is a switch rather than an addition: a server
        started that way stops answering chat. So it belongs to a separate
        entry with a dedicated model, not to a path advertised on entries that
        return 501 — which is what all eleven on the container did while the
        page said every model answered it.
        """
        self.assertNotIn("/v1/embeddings", OPENAI_PATHS)
        self.assertNotIn("/v1/embeddings", ANTHROPIC_PATHS)

    def test_what_is_offered_needs_no_extra_flag(self):
        # Verified on the container against models started with the ordinary
        # settings: chat and completions on both engines, messages and
        # count_tokens on vLLM.
        self.assertEqual(OPENAI_PATHS,
                         ("/v1/chat/completions", "/v1/completions"))
        self.assertEqual(ANTHROPIC_PATHS,
                         ("/v1/messages", "/v1/messages/count_tokens"))
