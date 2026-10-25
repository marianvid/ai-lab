"""A request turned into the ComfyUI graph of its profile."""

import json
import tempfile
import unittest
from pathlib import Path

from ai_lab.media.image_workflows import INPUT_MARKER, read_request, render
from ai_lab.multipart import MultipartBody

GRAPH = {"1": {"class_type": "Prompt", "inputs": {"text": "", "seed": 0}},
         "2": {"class_type": "LoadImage", "inputs": {"image": ""}}}
PROFILE = {"workflow": "edit.json",
           "inputs": {"prompt": ["1", "text"], "seed": ["1", "seed"], "image": ["2", "image"]}}


class RenderTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        (self.root / "edit.json").write_text(json.dumps(GRAPH))

    def test_mapped_values_and_the_image_marker_reach_their_inputs(self):
        graph = render(PROFILE, {"prompt": "a cat", "seed": 7, "steps": 4}, self.root)
        self.assertEqual(graph["1"]["inputs"], {"text": "a cat", "seed": 7})
        self.assertEqual(graph["2"]["inputs"]["image"], INPUT_MARKER)

    def test_a_mapping_to_a_missing_input_is_refused(self):
        profile = {**PROFILE, "inputs": {"prompt": ["9", "text"]}}
        with self.assertRaisesRegex(ValueError, "targets missing 9.text"):
            render(profile, {"prompt": "x"}, self.root)

    def test_a_workflow_outside_the_root_is_refused(self):
        with self.assertRaisesRegex(ValueError, "escapes workflow_root"):
            render({"workflow": "../edit.json"}, {}, self.root)


class ReadRequestTests(unittest.TestCase):
    def test_json_is_taken_as_it_is(self):
        self.assertEqual(read_request({"prompt": "x"}), ({"prompt": "x"}, None))

    def test_form_fields_get_the_types_json_would_carry(self):
        boundary = "b"
        raw = "".join(f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n'
                      f"{value}\r\n" for key, value in
                      (("seed", "3"), ("cfg_scale", "1.5"), ("prompt", "dog"))).encode() + f"--{boundary}--\r\n".encode()
        fields, image = read_request(MultipartBody(f"multipart/form-data; boundary={boundary}", raw))
        self.assertEqual(fields, {"seed": 3, "cfg_scale": 1.5, "prompt": "dog"})
        self.assertIsNone(image)

    def test_anything_else_is_refused(self):
        with self.assertRaisesRegex(ValueError, "JSON or multipart"):
            read_request([1, 2])
