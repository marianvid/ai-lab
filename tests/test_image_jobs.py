import json
import tempfile
import unittest
from pathlib import Path

from ai_lab.images.jobs import ImageJobs
from ai_lab.types import Task


class Gateway:
    max_upload_bytes = 100000
    max_upload_pixels = 100000
    max_upload_dimension = 1000


class ImageJobProfileTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        (root / "workflow.json").write_text(json.dumps({
            "1": {"class_type": "CLIPTextEncode", "inputs": {"text": ""}},
            "2": {"class_type": "KSampler", "inputs": {"seed": 0}},
        }))
        self.jobs = ImageJobs(Gateway(), {
            "workflow_root": str(root),
            "state_dir": str(root / "state"),
            "profiles": {"safe": {"model": "image", "workflow": "workflow.json",
                                    "task": "generation",
                                    "inputs": {"prompt": ["1", "text"],
                                               "seed": ["2", "seed"]}}}}, root)

    def tearDown(self):
        self.temp.cleanup()

    def test_only_a_named_profile_can_render(self):
        graph = self.jobs._render(self.jobs.settings["profiles"]["safe"],
                                  {"prompt": "panel", "seed": 42})
        self.assertEqual(graph["1"]["inputs"]["text"], "panel")
        self.assertEqual(graph["2"]["inputs"]["seed"], 42)

    def test_workflow_cannot_escape_root(self):
        with self.assertRaises(ValueError):
            self.jobs._render({"workflow": "../outside.json"}, {})

    def test_restart_fails_unfinished_job_without_prompt_data(self):
        job = {"id": "unfinished", "profile": "safe", "model": "image",
               "task": Task.IMAGE_GENERATION.value, "status": "running",
               "created_at": 1, "updated_at": 1, "error": "", "result": None}
        self.jobs._save(job)
        recovered = ImageJobs(Gateway(), self.jobs.settings,
                              Path(self.temp.name)).get("unfinished")
        self.assertEqual(recovered["status"], "failed")
        self.assertNotIn("prompt", recovered)

    def test_list_does_not_embed_result_bytes(self):
        job = {"id": "complete", "profile": "safe", "model": "image",
               "task": Task.IMAGE_GENERATION.value, "status": "succeeded",
               "created_at": 1, "updated_at": 1, "error": "",
               "result": {"data": [{"id": "image-0", "mime_type": "image/png",
                                      "b64_json": "many-bytes"}]}}
        self.jobs.jobs[job["id"]] = job
        self.assertNotIn("b64_json", self.jobs.list()[0]["result"]["data"][0])


if __name__ == "__main__":
    unittest.main()
