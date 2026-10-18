import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ai_lab.images.comfyui_server import Backend, ComfyLaunch, replace_marker


class ComfyUiServerTests(unittest.TestCase):
    def command(self, mode, workflow=None, ui_workflow=None):
        with tempfile.TemporaryDirectory() as directory, \
             patch("ai_lab.images.comfyui_server.subprocess.Popen") as popen:
            Backend(ComfyLaunch("python", "main.py", ["/models"], Path(directory), 18114,
                                vram_mode=mode, workflow=workflow, ui_workflow=ui_workflow),
                    wait_ready=lambda backend: None)
            return popen.call_args.args[0], popen.call_args.kwargs, Path(directory, "extra_model_paths.yaml").read_text()

    def test_low_vram_reaches_comfyui(self):
        self.assertIn("--lowvram", self.command("low")[0])

    def test_normal_mode_adds_no_memory_override(self):
        command = self.command("normal")[0]
        self.assertNotIn("--lowvram", command)
        self.assertNotIn("--cpu", command)

    def test_private_backend_uses_its_assigned_port(self):
        command = self.command("normal")[0]
        self.assertEqual(command[command.index("--port") + 1], "18114")

    def test_native_ui_is_network_accessible_and_loads_only_its_preset(self):
        command, options, paths = self.command("normal", Path("/workflows/one.json"))
        self.assertEqual(command[command.index("--listen") + 1], "0.0.0.0")
        self.assertIn("custom_nodes: .", paths)
        self.assertEqual(options["env"]["AI_LAB_COMFYUI_PRESET"], "/workflows/one.json")

    def test_no_preset_environment_when_one_is_not_configured(self):
        _, options, _ = self.command("normal")
        self.assertNotIn("AI_LAB_COMFYUI_PRESET", options["env"])
        self.assertNotIn("AI_LAB_COMFYUI_UI_PRESET", options["env"])

    def test_native_template_is_separate_from_generation_workflow(self):
        _, options, _ = self.command("normal", Path("/workflows/api.json"),
                                     Path("/workflows/native.json"))
        self.assertEqual(options["env"]["AI_LAB_COMFYUI_PRESET"],
                         "/workflows/api.json")
        self.assertEqual(options["env"]["AI_LAB_COMFYUI_UI_PRESET"],
                         "/workflows/native.json")

    def test_markers_are_replaced_wherever_they_are_nested(self):
        workflow = {"a": "__X__", "b": [{"c": "__X__"}, "keep"], "d": 3}
        replace_marker(workflow, "__X__", "done")
        self.assertEqual(workflow, {"a": "done", "b": [{"c": "done"}, "keep"], "d": 3})


if __name__ == "__main__":
    unittest.main()
