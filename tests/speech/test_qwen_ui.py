"""Qwen's native editor must share the already-loaded model."""

import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from ai_lab.speech.server import launch_native_qwen_ui


class QwenNativeUiTests(unittest.TestCase):
    def test_editor_reuses_model_and_binds_companion_port(self):
        seen = {}
        demo = types.SimpleNamespace(
            queue=lambda **kwargs: seen.setdefault("queue", kwargs),
            launch=lambda **kwargs: seen.setdefault("launch", kwargs),
        )
        package = types.ModuleType("qwen_tts")
        package.__path__ = []
        cli = types.ModuleType("qwen_tts.cli")
        cli.__path__ = []
        module = types.ModuleType("qwen_tts.cli.demo")
        module.build_demo = lambda *args: seen.setdefault("build", args) and demo
        backend = types.SimpleNamespace(model=object())

        with patch.dict(sys.modules, {"qwen_tts": package,
                                      "qwen_tts.cli": cli,
                                      "qwen_tts.cli.demo": module}):
            launch_native_qwen_ui(backend, Path("/models/qwen"), 18113)

        self.assertIs(seen["build"][0], backend.model)
        self.assertEqual(seen["build"][1], "/models/qwen")
        self.assertEqual(seen["launch"]["server_port"], 18113)
        self.assertFalse(seen["launch"]["share"])
        self.assertEqual(seen["queue"]["default_concurrency_limit"], 1)


if __name__ == "__main__":
    unittest.main()
