import unittest

from ai_lab.engines.stableaudio3 import StableAudio3Engine
from ai_lab.types import Format, ModelSet, Task


def model(name="stable-audio-3-medium"):
    return ModelSet(id=f"audio-music/{name}", name=name, format=Format.SAFETENSORS,
                    entrypoint=f"/models/audio/music/{name}", files=(),
                    task=Task.MUSIC_GENERATION)


class StableAudio3EngineTests(unittest.TestCase):
    def setUp(self):
        self.engine = StableAudio3Engine(binary="/py", model_options={
            "stable-audio-3-medium": {"steps": 8, "cfg_scale": 1.0,
                                      "memory_reservation_mb": 9000}})

    def test_plan_runs_offline_from_the_model_folder(self):
        plan = self.engine.plan(model(), 8131, {})
        self.assertEqual(plan.argv[plan.argv.index("--model-path") + 1],
                         "/models/audio/music/stable-audio-3-medium")
        self.assertEqual(plan.env["HF_HUB_OFFLINE"], "1")

    def test_instance_settings_are_refused(self):
        with self.assertRaises(ValueError):
            self.engine.plan(model(), 8131, {"steps": 4})
