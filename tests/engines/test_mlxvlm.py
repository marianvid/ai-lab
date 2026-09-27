import asyncio
import json
import sys
import unittest
from pathlib import Path
from unittest import mock

from ai_lab.engines import base
from ai_lab.engines.base import withheld
from ai_lab.engines.mlxvlm import MlxVlmEngine, default_launcher
from ai_lab.engines.registry import Registry
from ai_lab.text import mlxvlm_server as launcher
from ai_lab.types import Format, ModelFile, ModelSet, Task

PYTHON = "/runtime/mlxvlm/current/bin/python"


def model(format=Format.MLX, complete=True, task=Task.TEXT_GENERATION,
          size=30 * 1024 * 1024 * 1024):
    entrypoint = "/models/mlx/qwen-4bit"
    return ModelSet(id="mlx/qwen-4bit", name="qwen-4bit", format=format,
                    entrypoint=entrypoint,
                    files=(ModelFile(entrypoint + "/model.safetensors", size),),
                    task=task, complete=complete,
                    missing=() if complete else ("model-00002-of-00002",),
                    capabilities=frozenset({"images", "tools"}))


class PlanTests(unittest.TestCase):
    def setUp(self):
        self.engine = MlxVlmEngine(binary=PYTHON, server="/app/mlxvlm_server.py")

    def argv(self, params=None):
        return self.engine.plan(model(), 8150, params or {}).argv

    def flag(self, name, params=None):
        argv = self.argv(params)
        return argv[argv.index(name) + 1]

    def defaults(self, params=None):
        argv = self.argv(params)
        pairs = [argv[i + 1] for i, item in enumerate(argv) if item == "--request-default"]
        return {key: json.loads(value) for key, _, value in
                (pair.partition("=") for pair in pairs)}

    def test_the_launcher_runs_in_the_engine_environment_with_the_folder(self):
        argv = self.argv()
        self.assertEqual(argv[:2], [PYTHON, "/app/mlxvlm_server.py"])
        self.assertEqual(self.flag("--model"), "/models/mlx/qwen-4bit")
        self.assertEqual(self.flag("--port"), "8150")

    def test_defaults_are_applied(self):
        self.assertEqual(self.flag("--max-num-seqs"), "8")
        self.assertEqual(self.flag("--prefill-step-size"), "2048")
        self.assertEqual(self.flag("--vision-cache-size"), "20")
        self.assertEqual(self.flag("--max-tokens"), "8192")
        self.assertNotIn("--max-kv-size", self.argv())

    def test_sampling_defaults_travel_as_request_defaults(self):
        # mlx-vlm has no flags for these; the launcher fills them in.
        self.assertEqual(self.defaults(), {"temperature": 0.8, "top_p": 0.95,
                                           "top_k": 40, "min_p": 0.05})
        self.assertEqual(self.defaults({"temperature": 0.2})["temperature"], 0.2)

    def test_thinking_is_on_unless_switched_off(self):
        # mlx-vlm on its own turns thinking off; auto matches the templates.
        self.assertIn("--enable-thinking", self.argv())
        self.assertIn("--enable-thinking", self.argv({"reasoning": "on"}))
        self.assertNotIn("--enable-thinking", self.argv({"reasoning": "off"}))

    def test_settings_reach_the_command_line(self):
        self.assertEqual(self.flag("--max-num-seqs", {"max_num_seqs": 4}), "4")
        self.assertEqual(self.flag("--max-kv-size", {"max_kv_size": 36864}), "36864")

    def test_invalid_and_foreign_settings_are_refused(self):
        for bad in ({"max_num_seqs": 0}, {"reasoning": "maybe"},
                    {"decode_concurrency": 4}):
            with self.assertRaises(ValueError):
                self.argv(bad)

    def test_other_formats_tasks_and_incomplete_models_are_refused(self):
        for wrong in (model(format=Format.GGUF), model(task=Task.TRANSCRIPTION),
                      model(complete=False)):
            with self.assertRaises(ValueError):
                self.engine.plan(wrong, 8150, {})


class DescriptionTests(unittest.TestCase):
    def setUp(self):
        self.engine = MlxVlmEngine(binary=PYTHON)

    def test_it_reads_mlx_and_keeps_pictures(self):
        self.assertEqual(self.engine.formats(), frozenset({Format.MLX}))
        self.assertEqual(withheld(self.engine, {}), frozenset())

    def test_it_answers_the_openai_shape(self):
        self.assertEqual(self.engine.api_paths(), base.OPENAI_PATHS)

    def test_ready_only_once_a_model_is_named(self):
        with mock.patch("ai_lab.engines.mlxvlm.http_json",
                        return_value={"status": "healthy", "loaded_model": None}):
            self.assertFalse(self.engine.ready(8150))
        with mock.patch("ai_lab.engines.mlxvlm.http_json",
                        return_value={"status": "healthy", "loaded_model": "/m"}):
            self.assertTrue(self.engine.ready(8150))
        with mock.patch("ai_lab.engines.mlxvlm.http_json", return_value={}):
            self.assertFalse(self.engine.ready(8150))

    def test_memory_and_concurrency(self):
        self.assertAlmostEqual(self.engine.needs_mb(model(), {}, 0), 30 * 1024 * 1.1)
        self.assertEqual(self.engine.concurrency({"max_num_seqs": 4}), 4)

    def test_the_default_launcher_ships_with_the_application(self):
        self.assertTrue(Path(default_launcher()).is_file())

    def test_the_registry_builds_it_from_configuration(self):
        engine = Registry({"mlxvlm": {"binary": PYTHON}}).get("mlxvlm")
        self.assertEqual(engine.plan(model(), 8150, {}).argv[0], PYTHON)


class LauncherTests(unittest.TestCase):
    """The request adjustments made before mlx-vlm sees a request."""

    def test_its_own_options_are_taken_out(self):
        defaults, rest = launcher.split_arguments(
            ["--model", "/m", "--request-default", "top_k=40",
             "--request-default", "temperature=0.8", "--port", "1"])
        self.assertEqual(defaults, {"top_k": 40, "temperature": 0.8})
        self.assertEqual(rest, ["--model", "/m", "--port", "1"])
        self.assertEqual(launcher.model_of(rest), "/m")

    def test_any_model_name_means_the_loaded_folder(self):
        self.assertEqual(launcher.adjust({"model": "qwen-4bit"}, "/m", {})["model"], "/m")

    def test_the_shared_thinking_field_is_understood(self):
        body = {"chat_template_kwargs": {"enable_thinking": False}}
        self.assertIs(launcher.adjust(body, "/m", {})["enable_thinking"], False)
        own = {"enable_thinking": True, "chat_template_kwargs": {"enable_thinking": False}}
        self.assertIs(launcher.adjust(own, "/m", {})["enable_thinking"], True,
                      "mlx-vlm's own field, when sent, wins")

    def test_defaults_fill_only_what_is_missing(self):
        body = launcher.adjust({"temperature": 0}, "/m", {"temperature": 0.8, "top_k": 40})
        self.assertEqual((body["temperature"], body["top_k"]), (0, 40))

    def test_picture_parts_pass_through_untouched(self):
        parts = [{"type": "text", "text": "what is this?"},
                 {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]
        body = launcher.adjust({"messages": [{"role": "user", "content": parts}]}, "/m", {})
        self.assertEqual(body["messages"][0]["content"], parts)

    def test_the_middleware_rewrites_the_body_and_its_length(self):
        seen = {}

        async def app(scope, receive, send):
            message = await receive()
            seen["body"] = json.loads(message["body"])
            seen["length"] = dict(scope["headers"])[b"content-length"]

        raw = json.dumps({"model": "x", "messages": []}).encode()
        messages = [{"type": "http.request", "body": raw[:5], "more_body": True},
                    {"type": "http.request", "body": raw[5:], "more_body": False}]

        async def receive():
            return messages.pop(0)

        scope = {"type": "http", "method": "POST",
                 "headers": [(b"content-type", b"application/json"),
                             (b"content-length", str(len(raw)).encode())]}
        middleware = launcher.AdjustRequests(app, model="/m", defaults={"top_k": 40})
        asyncio.run(middleware(scope, receive, None))
        self.assertEqual(seen["body"], {"model": "/m", "messages": [], "top_k": 40})
        self.assertEqual(int(seen["length"]), len(json.dumps(seen["body"]).encode()))


@unittest.skipUnless(sys.platform == "darwin", "macOS only")
class HostTests(unittest.TestCase):
    def test_the_mac_supports_it(self):
        from ai_lab.hosts.darwin import DarwinHost
        self.assertIn("mlxvlm", DarwinHost().capabilities().supported_engines)


class LinuxTests(unittest.TestCase):
    def test_linux_never_offers_it(self):
        from ai_lab.hosts import linux
        self.assertNotIn("mlxvlm", Path(linux.__file__).read_text())


if __name__ == "__main__":
    unittest.main()
