import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ai_lab.images.comfyui_server import Backend


class ComfyUiServerTests(unittest.TestCase):
    def command(self, mode):
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(Backend, "_wait_ready"), \
             patch("ai_lab.images.comfyui_server.subprocess.Popen") as popen:
            Backend("python", "main.py", ["/models"], Path(directory),
                    vram_mode=mode)
            return popen.call_args.args[0]

    def test_low_vram_reaches_comfyui(self):
        self.assertIn("--lowvram", self.command("low"))

    def test_normal_mode_adds_no_memory_override(self):
        command = self.command("normal")
        self.assertNotIn("--lowvram", command)
        self.assertNotIn("--cpu", command)


if __name__ == "__main__":
    unittest.main()
