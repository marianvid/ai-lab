"""VoxCPM's upstream editor must reuse the loaded checkpoint."""

import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from ai_lab.speech.server import launch_native_voxcpm_ui


class VoxCpmNativeUiTests(unittest.TestCase):
    def test_editor_reuses_model_and_binds_companion_port(self):
        seen = {}

        class Demo:
            def __init__(self, model_id):
                seen["model_id"] = model_id
                self.voxcpm_model = None

        interface = types.SimpleNamespace(
            queue=lambda **kwargs: seen.setdefault("queue", kwargs) and interface,
            launch=lambda **kwargs: seen.setdefault("launch", kwargs))
        module = types.ModuleType("ai_lab.native_ui.voxcpm_upstream.app")
        module.VoxCPMDemo = Demo
        module.create_demo_interface = lambda demo: seen.setdefault("demo", demo) and interface
        module.I18N = object()
        module._APP_THEME = object()
        module._CUSTOM_CSS = ""
        backend = types.SimpleNamespace(model=object())

        with patch.dict(sys.modules, {"ai_lab.native_ui.voxcpm_upstream.app": module}):
            launch_native_voxcpm_ui(backend, Path("/models/voxcpm"), 18113)

        self.assertEqual(seen["model_id"], "/models/voxcpm")
        self.assertIs(seen["demo"].voxcpm_model, backend.model)
        self.assertEqual(seen["launch"]["server_port"], 18113)
        self.assertFalse(seen["launch"]["share"])
        self.assertEqual(seen["queue"]["default_concurrency_limit"], 1)


if __name__ == "__main__":
    unittest.main()
