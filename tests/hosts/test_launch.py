import json
import unittest
from unittest import mock
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
                path = launch.write_spec(ProcessSpec(good, ["x"], {}), self.directory)
                self.assertTrue(path.is_file())

    def test_the_file_is_plain_json_the_launcher_can_read(self):
        path = launch.write_spec(ProcessSpec("qwen", ["a", "b"], {}), self.directory)
        self.assertEqual(json.loads(path.read_text())["argv"], ["a", "b"])


class StateDirTests(unittest.TestCase):
    def test_the_directory_comes_from_the_environment(self):
        with mock.patch.dict("os.environ", {launch.STATE_DIR_VARIABLE: "/srv/lab"}):
            self.assertEqual(launch.state_dir(), Path("/srv/lab"))
            self.assertEqual(launch.launch_dir(), Path("/srv/lab/launch"))

    def test_an_unset_variable_fails_loudly(self):
        with mock.patch.dict("os.environ", {launch.STATE_DIR_VARIABLE: ""}), \
                self.assertRaisesRegex(RuntimeError, launch.STATE_DIR_VARIABLE):
            launch.state_dir()

    def test_the_default_directory_is_used_without_an_argument(self):
        with TemporaryDirectory() as root, \
                mock.patch.dict("os.environ", {launch.STATE_DIR_VARIABLE: root}):
            launch.write_spec(ProcessSpec(instance_id="x1", argv=["a"], env={}))
            self.assertEqual(launch.read_spec("x1"), (["a"], {}))


class MergedEnvTests(unittest.TestCase):
    """A plan's PATH prefix goes before the starting unit's own PATH."""

    def test_the_prefix_goes_before_the_units_path(self):
        env = launch.merged_env({"AI_LAB_PATH_PREPEND": "/usr/local/cuda/bin", "X": "1"},
                                base={"PATH": "/usr/bin:/bin", "HOME": "/h"})
        self.assertEqual(env["PATH"], "/usr/local/cuda/bin:/usr/bin:/bin")
        self.assertEqual(env["X"], "1")
        self.assertEqual(env["HOME"], "/h")
        self.assertNotIn("AI_LAB_PATH_PREPEND", env)

    def test_no_prefix_changes_nothing(self):
        env = launch.merged_env({"X": "1"}, base={"PATH": "/usr/bin"})
        self.assertEqual(env, {"PATH": "/usr/bin", "X": "1"})

    def test_a_prefix_without_a_path_is_the_path(self):
        env = launch.merged_env({"AI_LAB_PATH_PREPEND": "/c/bin"}, base={})
        self.assertEqual(env["PATH"], "/c/bin")


if __name__ == "__main__":
    unittest.main()
