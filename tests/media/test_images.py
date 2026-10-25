"""An image call by named profile: checked, sent through the gateway, answered with the images."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ai_lab.media.images import ImageCalls
from ai_lab.types import Task

GRAPH = {"1": {"class_type": "Prompt", "inputs": {"text": ""}}}


class Lease:
    port = 8105

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class Gateway:
    max_upload_bytes = 0
    max_upload_pixels = 0
    max_upload_dimension = 0

    def __init__(self):
        self.asked = []

    def acquire(self, model, shape=None):
        self.asked.append((model, shape))
        return Lease()

    def timeouts_for(self, task):
        return (1800.0, 60.0)


class ImageCallsTest(unittest.TestCase):
    def setUp(self):
        root = Path(tempfile.mkdtemp())
        (root / "gen.json").write_text(json.dumps(GRAPH))
        self.gateway = Gateway()
        self.images = ImageCalls(self.gateway, {"workflow_root": str(root), "profiles": {
            "qwen": {"task": "generation", "model": "image-qwen", "workflow": "gen.json",
                     "inputs": {"prompt": ["1", "text"]}, "description": "Qwen"},
            "edit": {"task": "edit", "model": "image-qwen-edit", "workflow": "gen.json",
                     "inputs": {}}}})

    def answer(self, payload):
        @contextlib.contextmanager
        def opened(request, timeout):
            self.sent = (request.full_url, json.loads(request.data), timeout)
            yield io.BytesIO(json.dumps(payload).encode())
        return mock.patch("urllib.request.urlopen", opened)

    def test_profiles_are_listed(self):
        self.assertEqual(self.images.profiles()[1],
                         {"id": "qwen", "task": "generation", "model": "image-qwen",
                          "description": "Qwen"})

    def test_a_generation_goes_through_the_gateway_and_returns_the_images(self):
        with self.answer({"data": [{"b64_json": "AAAA"}]}):
            made = self.images.submit({"profile": "qwen", "prompt": "a cat"},
                                      Task.IMAGE_GENERATION)
        self.assertEqual(made["data"], [{"b64_json": "AAAA"}])
        self.assertEqual(self.gateway.asked, [("image-qwen", "/v1/images/generations")])
        url, work, timeout = self.sent
        self.assertEqual(url, "http://127.0.0.1:8105/v1/images/generations")
        self.assertEqual(work["workflow"]["1"]["inputs"]["text"], "a cat")
        self.assertEqual(timeout, 1800.0)

    def test_an_engine_without_images_is_an_error(self):
        with self.answer({"data": []}), self.assertRaisesRegex(RuntimeError, "no images"):
            self.images.submit({"profile": "qwen"}, Task.IMAGE_GENERATION)

    def test_wrong_requests_are_refused_before_the_card(self):
        with self.assertRaisesRegex(ValueError, "Unknown image fields: async"):
            self.images.submit({"profile": "qwen", "async": True}, Task.IMAGE_GENERATION)
        with self.assertRaisesRegex(ValueError, "Unknown image profile"):
            self.images.submit({"profile": "nope"}, Task.IMAGE_GENERATION)
        with self.assertRaisesRegex(ValueError, "does not support edit"):
            self.images.submit({"profile": "qwen"}, Task.IMAGE_EDIT)
        with self.assertRaisesRegex(ValueError, "require an image"):
            self.images.submit({"profile": "edit"}, Task.IMAGE_EDIT)
        self.assertEqual(self.gateway.asked, [])


if __name__ == "__main__":
    unittest.main()
