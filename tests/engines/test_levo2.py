import unittest

from ai_lab.engines.levo2 import Levo2Engine
from ai_lab.types import Format, ModelSet, Task

OPTIONS = {"levo2-gguf": {"lm": "LeVo2-v2-large-F16.gguf",
                          "flow": "LeVo2-v2-flow-F32.gguf",
                          "vae": "LeVo2-v2-vae-F32.gguf", "steps": 50,
                          "cfg": 1.5, "memory_reservation_mb": 20000}}


def model(name="levo2-gguf"):
    return ModelSet(id=f"audio-music/{name}", name=name, format=Format.SAFETENSORS,
                    entrypoint=f"/models/audio/music/{name}", files=(),
                    task=Task.MUSIC_GENERATION)


class Levo2EngineTests(unittest.TestCase):
    def setUp(self):
        self.engine = Levo2Engine(binary="/py", cantor="/opt/levo-cantor",
                                  output_root="/var/jobs", model_options=OPTIONS)

    def test_plan_passes_the_three_files_and_render_settings(self):
        argv = self.engine.plan(model(), 8130, {}).argv
        for flag, value in (("--cantor", "/opt/levo-cantor"),
                            ("--lm", "LeVo2-v2-large-F16.gguf"),
                            ("--flow", "LeVo2-v2-flow-F32.gguf"),
                            ("--vae", "LeVo2-v2-vae-F32.gguf"),
                            ("--steps", "50"), ("--port", "8130")):
            self.assertEqual(argv[argv.index(flag) + 1], value)

    def test_only_configured_models_are_served(self):
        self.assertTrue(self.engine.supports(model()))
        self.assertFalse(self.engine.supports(model("other")))
        self.assertEqual(self.engine.needs_mb(model(), {}, 32000), 20000)
