import unittest

from ai_lab.api.router import Router


class RouterTests(unittest.TestCase):
    def setUp(self):
        self.router = Router()
        self.router.add("GET", "/api/models", lambda **_: "models")
        self.router.add("POST", "/api/instances/{id}/load", lambda **kw: kw)

    def test_a_literal_path_matches(self):
        handler, captured = self.router.match("GET", "/api/models")
        self.assertEqual(handler(), "models")
        self.assertEqual(captured, {})

    def test_a_placeholder_is_captured(self):
        _, captured = self.router.match("POST", "/api/instances/qwen/load")
        self.assertEqual(captured, {"id": "qwen"})

    def test_encoded_segments_are_decoded(self):
        """Model ids contain slashes, so the browser sends them encoded."""
        self.router.add("GET", "/api/models/{model_id}", lambda **kw: kw)
        _, captured = self.router.match("GET", "/api/models/repo%2Fqwen%2Fmodel")
        self.assertEqual(captured, {"model_id": "repo/qwen/model"})

    def test_the_method_must_match(self):
        self.assertIsNone(self.router.match("POST", "/api/models"))

    def test_an_unknown_path_does_not_match(self):
        self.assertIsNone(self.router.match("GET", "/api/nope"))

    def test_length_must_match(self):
        self.assertIsNone(self.router.match("GET", "/api/models/extra"))


if __name__ == "__main__":
    unittest.main()
