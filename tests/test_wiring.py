"""What the composition root works out from the configuration before building anything."""

import tempfile
import unittest
from pathlib import Path

from ai_lab.config import Config, Instance
from ai_lab.wiring import engine_settings_of, gateway_limits


class EngineSettingsTests(unittest.TestCase):
    def test_comfyui_is_told_each_image_model_s_workflow(self):
        root = Path(tempfile.mkdtemp())
        config = Config(
            engines={"comfyui": {"binary": "/opt/python"}, "llamacpp": {"binary": "x"}},
            instances=[Instance(id="flux", engine="comfyui", model_id="comfy/flux", port=8200)],
            images={"workflow_root": str(root),
                    "profiles": {"draw": {"model": "flux", "workflow": "flux.json"},
                                 "outside": {"model": "flux", "workflow": "../escape.json"}}})
        settings = engine_settings_of(config)
        self.assertEqual(settings["llamacpp"], {"binary": "x"})
        self.assertEqual(settings["comfyui"]["binary"], "/opt/python")
        self.assertEqual(settings["comfyui"]["preset_workflows"],
                         {"comfy/flux": str((root / "flux.json").resolve())})

    def test_a_workflow_outside_the_workflow_root_is_ignored(self):
        root = Path(tempfile.mkdtemp())
        config = Config(
            instances=[Instance(id="flux", engine="comfyui", model_id="comfy/flux", port=8200)],
            images={"workflow_root": str(root),
                    "profiles": {"outside": {"model": "flux", "workflow": "../escape.json"}}})
        self.assertEqual(engine_settings_of(config)["comfyui"]["preset_workflows"], {})


class GatewayLimitsTests(unittest.TestCase):
    def test_the_front_door_s_limits_come_from_the_configuration(self):
        limits = gateway_limits(Config(gateway={"first_byte_s": 60, "max_waiting": 12}))
        self.assertEqual((limits.first_byte_s, limits.max_waiting), (60, 12))
