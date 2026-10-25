"""What the composition root works out from the configuration before building anything."""

import tempfile
import unittest
from pathlib import Path

from ai_lab.config.store import Config, Instance
from ai_lab.wiring import CONFIG_SECTIONS, engine_settings_of, gateway_limits


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


class SectionsTests(unittest.TestCase):
    def test_the_sections_other_packages_own_are_checked_by_their_parsers(self):
        self.assertEqual(sorted(CONFIG_SECTIONS),
                         ["cpu_models", "gpu_health", "providers"])
        for name in ("gpu_health",):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "Unknown"):
                CONFIG_SECTIONS[name].parse({"nope": 1})

    def test_an_embeddings_model_claims_its_ports(self):
        section = CONFIG_SECTIONS["cpu_models"]
        parsed = section.parse({"binary": "/bin/llama-server", "models": {"bge-m3": {
            "path": "/m/bge.gguf", "port": 8390, "card": {"port": 8391, "needs_mb": 1300}}}})
        assert section.ports is not None
        self.assertEqual(list(section.ports(parsed)),
                         [("cpu model bge-m3", 8390), ("cpu model bge-m3", 8391)])
