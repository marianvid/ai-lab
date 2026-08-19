import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from ai_lab.hosts import launch
from ai_lab.types import ProcessSpec


class LaunchSpecTests(unittest.TestCase):
    def setUp(self):
        self._temporary = TemporaryDirectory()
        self.directory = Path(self._temporary.name)
        self.addCleanup(self._temporary.cleanup)

    def test_a_command_survives_the_round_trip(self):
        launch.write_spec(ProcessSpec("qwen", ["llama-server", "--model", "/a.gguf"],
                                      {"CUDA_VISIBLE_DEVICES": "0"}), self.directory)
        argv, env = launch.read_spec("qwen", self.directory)
        self.assertEqual(argv, ["llama-server", "--model", "/a.gguf"])
        self.assertEqual(env, {"CUDA_VISIBLE_DEVICES": "0"})

    def test_writing_is_atomic(self):
        launch.write_spec(ProcessSpec("qwen", ["x"], {}), self.directory)
        self.assertEqual([item.name for item in self.directory.iterdir()], ["qwen.json"])

    def test_a_dangerous_instance_id_is_refused(self):
        """The id becomes part of a file path and a systemd unit name."""
        for bad in ("../escape", "qwen; rm -rf /", "Qwen", "", "qwen/../x"):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    launch.write_spec(ProcessSpec(bad, ["x"], {}), self.directory)
                with self.assertRaises(ValueError):
                    launch.read_spec(bad, self.directory)

    def test_a_plain_id_is_accepted(self):
        for good in ("qwen", "qwen-coder", "a1", "gemma-general-2"):
            with self.subTest(good=good):
                launch.write_spec(ProcessSpec(good, ["x"], {}), self.directory)

    def test_the_file_is_plain_json_the_launcher_can_read(self):
        path = launch.write_spec(ProcessSpec("qwen", ["a", "b"], {}), self.directory)
        self.assertEqual(json.loads(path.read_text())["argv"], ["a", "b"])


if __name__ == "__main__":
    unittest.main()
