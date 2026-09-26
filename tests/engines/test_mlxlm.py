import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from ai_lab.catalog import Catalog
from ai_lab.config import Repository
from ai_lab.engines import base
from ai_lab.engines.mlxlm import MlxLmEngine, default_launcher
from ai_lab.engines.registry import Registry
from ai_lab.naming import is_weight_for_format
from ai_lab.types import Capabilities, Format, ModelFile, ModelSet, Task

PYTHON = "/runtime/mlxlm/current/bin/python"


def model(format=Format.MLX, complete=True, task=Task.TEXT_GENERATION,
          size=10 * 1024 * 1024 * 1024):
    entrypoint = "/models/mlx/qwen-4bit"
    return ModelSet(id="mlx/qwen-4bit", name="qwen-4bit", format=format,
                    entrypoint=entrypoint,
                    files=(ModelFile(entrypoint + "/model.safetensors", size),),
                    task=task, complete=complete,
                    missing=() if complete else ("model-00002-of-00002",))


class PlanTests(unittest.TestCase):
    def setUp(self):
        self.engine = MlxLmEngine(binary=PYTHON, server="/app/mlxlm_server.py")

    def argv(self, params=None):
        return self.engine.plan(model(), 8130, params or {}).argv

    def flag(self, name, params=None):
        argv = self.argv(params)
        return argv[argv.index(name) + 1]

    def test_the_launcher_runs_in_the_engine_environment_with_the_folder(self):
        argv = self.argv()
        self.assertEqual(argv[:2], [PYTHON, "/app/mlxlm_server.py"])
        self.assertEqual(self.flag("--model"), "/models/mlx/qwen-4bit")
        self.assertEqual(self.flag("--port"), "8130")
        self.assertEqual(self.flag("--host"), "0.0.0.0")

    def test_defaults_are_applied(self):
        self.assertEqual(self.flag("--decode-concurrency"), "8")
        self.assertEqual(self.flag("--prompt-concurrency"), "4")
        self.assertEqual(self.flag("--prefill-step-size"), "2048")
        self.assertEqual(self.flag("--prompt-cache-size"), "10")
        self.assertEqual(self.flag("--max-tokens"), "8192")
        self.assertEqual(self.flag("--temp"), "0.8")
        self.assertEqual(self.flag("--top-p"), "0.95")
        self.assertEqual(self.flag("--top-k"), "40")
        self.assertEqual(self.flag("--min-p"), "0.05")

    def test_optional_flags_are_left_off_by_default(self):
        argv = self.argv()
        self.assertNotIn("--prompt-cache-bytes", argv)
        self.assertNotIn("--chat-template-args", argv)

    def test_settings_reach_the_command_line(self):
        self.assertEqual(self.flag("--decode-concurrency",
                                   {"decode_concurrency": 4}), "4")
        self.assertEqual(self.flag("--prompt-cache-bytes",
                                   {"prompt_cache_mb": 8192}), "8192MB")
        self.assertEqual(self.flag("--max-tokens", {"max_tokens": 100}), "100")

    def test_thinking_becomes_a_chat_template_switch(self):
        off = json.loads(self.flag("--chat-template-args", {"reasoning": "off"}))
        on = json.loads(self.flag("--chat-template-args", {"reasoning": "on"}))
        self.assertEqual(off, {"enable_thinking": False})
        self.assertEqual(on, {"enable_thinking": True})

    def test_invalid_settings_are_refused(self):
        for bad in ({"decode_concurrency": 0}, {"reasoning": "maybe"},
                    {"context_size": 32768}):
            with self.assertRaises(ValueError):
                self.argv(bad)

    def test_other_formats_and_tasks_are_refused(self):
        with self.assertRaises(ValueError):
            self.engine.plan(model(format=Format.GGUF), 8130, {})
        with self.assertRaises(ValueError):
            self.engine.plan(model(format=Format.SAFETENSORS), 8130, {})
        with self.assertRaises(ValueError):
            self.engine.plan(model(task=Task.TRANSCRIPTION), 8130, {})

    def test_an_incomplete_model_is_refused(self):
        with self.assertRaises(ValueError):
            self.engine.plan(model(complete=False), 8130, {})

    def test_the_launch_is_offline_and_unbuffered(self):
        env = self.engine.plan(model(), 8130, {}).env
        self.assertEqual(env["HF_HUB_OFFLINE"], "1")
        self.assertEqual(env["PYTHONUNBUFFERED"], "1")


class DescriptionTests(unittest.TestCase):
    def setUp(self):
        self.engine = MlxLmEngine(binary=PYTHON)

    def test_it_reads_mlx_and_generates_text(self):
        self.assertEqual(self.engine.formats(), frozenset({Format.MLX}))
        self.assertEqual(self.engine.tasks(), frozenset({Task.TEXT_GENERATION}))

    def test_it_answers_the_openai_shape_only(self):
        self.assertEqual(self.engine.api_paths(), base.OPENAI_PATHS)
        self.assertEqual(self.engine.api_paths(Task.TRANSCRIPTION), ())

    def test_readiness_asks_the_health_page(self):
        with mock.patch("ai_lab.engines.mlxlm.http_ok", return_value=True) as probe:
            self.assertTrue(self.engine.ready(8130))
        probe.assert_called_once_with(8130, "/health")

    def test_memory_is_the_weights_plus_a_margin(self):
        # A large model gets a tenth on top; a small one at least 2 GB.
        large = model(size=30 * 1024 * 1024 * 1024)
        self.assertAlmostEqual(self.engine.needs_mb(large, {}, 0), 30 * 1024 * 1.1)
        small = model(size=1024 * 1024 * 1024)
        self.assertEqual(self.engine.needs_mb(small, {}, 0), 1024 + 2048)

    def test_concurrency_is_the_decode_setting(self):
        self.assertEqual(self.engine.concurrency({}), 8)
        self.assertEqual(self.engine.concurrency({"decode_concurrency": 4}), 4)

    def test_the_default_launcher_ships_with_the_application(self):
        self.assertTrue(Path(default_launcher()).is_file())
        self.assertEqual(MlxLmEngine(binary=PYTHON).server, default_launcher())


class RegistryTests(unittest.TestCase):
    def test_the_configured_environment_reaches_the_command_line(self):
        registry = Registry({"mlxlm": {"binary": PYTHON}})
        argv = registry.get("mlxlm").plan(model(), 8130, {}).argv
        self.assertEqual(argv[0], PYTHON)

    def test_an_absent_runtime_is_reported_as_not_installed(self):
        rows = {row["id"]: row for row in Registry().describe(Capabilities(
            supervisor="subprocess", engines=frozenset(),
            accelerator_kind="metal"))}
        self.assertEqual(rows["mlxlm"]["reason"], "Not installed")


class FormatTests(unittest.TestCase):
    def test_mlx_weights_are_safetensors_files(self):
        self.assertTrue(is_weight_for_format("model-00001-of-00004.safetensors", "mlx"))
        self.assertFalse(is_weight_for_format("model.gguf", "mlx"))

    def test_an_mlx_folder_is_one_model_handed_over_as_a_folder(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / "qwen3.6-35b-a3b-4bit"
            folder.mkdir()
            for name in ("model-00001-of-00002.safetensors",
                         "model-00002-of-00002.safetensors",
                         "model.safetensors.index.json", "config.json",
                         "tokenizer.json"):
                (folder / name).write_bytes(b"x")
            models = Catalog().scan([Repository(id="mlx", name="MLX",
                                                path=str(root), format="mlx")])
        self.assertEqual(len(models), 1)
        self.assertEqual(models[0].id, "mlx/qwen3.6-35b-a3b-4bit")
        self.assertEqual(models[0].format, Format.MLX)
        self.assertEqual(models[0].entrypoint, str(folder))
        self.assertTrue(models[0].complete)


@unittest.skipUnless(sys.platform == "darwin", "macOS only")
class HostTests(unittest.TestCase):
    def test_the_mac_supports_it(self):
        from ai_lab.hosts.darwin import DarwinHost
        self.assertIn("mlxlm", DarwinHost().capabilities().supported_engines)


class LinuxTests(unittest.TestCase):
    def test_linux_never_offers_it(self):
        # MLX runs only on Apple silicon. The Linux host names every engine it
        # supports in its own file, so this one must not appear there.
        from ai_lab.hosts import linux
        self.assertNotIn("mlxlm", Path(linux.__file__).read_text())


if __name__ == "__main__":
    unittest.main()
