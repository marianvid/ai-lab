"""Each music server's `main`: what it builds from its command line, and what it serves."""

import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

from ai_lab.music import (
    comfy_server,
    heartmula_server,
    hosting,
    khala_server,
    levo2_server,
    mulacover_server,
    stableaudio3_server,
    yue2_server,
)


class Built:
    """Stands in for a backend class: remembers what it was built with."""

    made: list = []  # noqa: RUF012 - a record kept by the test, on purpose

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        Built.made.append(self)

    def close(self):
        """Nothing to stop."""


def run(module, argv, backend_name, **patches):
    """Run `module.main()` with `argv`; the backend built and how it was served."""
    Built.made = []
    served: list = []
    with ExitStack() as stack:
        stack.enter_context(patch.object(sys, "argv", ["server", *argv]))
        stack.enter_context(patch.object(module, backend_name, Built))
        stack.enter_context(patch.object(module, "serve", lambda *a, **k: served.append((a, k))))
        for name, value in patches.items():
            stack.enter_context(patch.object(module, name, value))
        module.main()
    return Built.made[0], served[0]


class ServerTests(unittest.TestCase):
    def test_levo2(self):
        backend, (args, _) = run(levo2_server, [
            "--cantor", "/c", "--model-path", "/m", "--lm", "lm", "--flow", "f",
            "--vae", "v", "--output-root", "/o", "--steps", "8", "--cfg", "1.5",
            "--port", "9000"], "Levo2Backend")
        files = backend.args[0]
        self.assertEqual((files.cantor, files.lm, backend.args[2]), (Path("/c"), "lm", 8))
        self.assertEqual(args[1], 9000)

    def test_mulacover_takes_a_large_body(self):
        backend, (_, kwargs) = run(mulacover_server, [
            "--bundle-root", "/b", "--checkpoint-path", "/b/c", "--checkpoint-subdir", "c",
            "--output-root", "/o", "--topk", "50", "--temperature", "1.0",
            "--cfg-scale", "1.5", "--port", "9001"], "MulaCoverBackend")
        self.assertEqual(backend.args[2].topk, 50)
        self.assertEqual(kwargs["max_body_bytes"], mulacover_server.MAX_BODY_BYTES)

    def test_heartmula_starts_its_page(self):
        pages = []
        backend, _ = run(heartmula_server, [
            "--bundle-root", "/b", "--checkpoint-path", "/b/c", "--checkpoint-subdir", "c",
            "--output-root", "/o", "--version", "3B", "--topk", "50",
            "--temperature", "1.0", "--cfg-scale", "1.5", "--port", "9002",
            "--ui-port", "9003", "--heartmuse-root", "/h"], "HeartMulaBackend",
            launch_heartmuse=lambda built, args: pages.append(args.ui_port))
        self.assertEqual((backend.args[2], pages), ("3B", [9003]))

    def test_stable_audio_starts_its_page(self):
        pages = []
        _, (_, kwargs) = run(stableaudio3_server, [
            "--model-path", "/m", "--steps", "8", "--cfg-scale", "1.0", "--port", "9004",
            "--ui-port", "9005"], "StableAudio3Backend",
            launch_native_ui=lambda built, port: pages.append(port))
        self.assertEqual((pages, kwargs["max_body_bytes"]),
                         ([9005], stableaudio3_server.MAX_BODY_BYTES))

    def test_khala_starts_its_app(self):
        apps = []
        backend, _ = run(khala_server, [
            "--generator-script", "/k/backend/gen.py", "--model-path", "/m",
            "--output-root", "/o", "--default-bucket", "1", "--maximum-bucket", "4",
            "--port", "9006", "--ui-port", "9007"], "KhalaBackend",
            launch_official_ui=lambda root, model, port: apps.append((root, port)))
        self.assertEqual((backend.args[3], apps), ((1, 4), [(Path("/k"), 9007)]))

    def test_comfy_runs_its_own_comfyui(self):
        with patch.object(comfy_server, "stop_on_exit", lambda stop: None):
            backend, _ = run(comfy_server, [
                "--comfyui", "/c", "--workflow", "/w.json", "--model-name", "minimax",
                "--ffmpeg", "ffmpeg", "--model-path", "/m1", "--model-path", "/m2",
                "--port", "9008"], "ComfyMusicBackend")
        runtime = backend.args[0]
        self.assertEqual((runtime.model_paths, runtime.port), (["/m1", "/m2"], 19008))

    def test_yue2_serves_through_its_studio(self):
        studios = []
        backend, _ = run(yue2_server, [
            "--model-path", "/m/yue2", "--vae-path", "/v", "--output-root", "/o",
            "--cot", "full", "--memory-budget-gib", "20", "--port", "9010",
            "--ui-port", "9011", "--webui-root", "/w"], "Yue2WebBackend",
            launch_studio=lambda args, url: studios.append(url))
        self.assertEqual(backend.args[1:], ("yue2", "full"))
        self.assertTrue(studios[0].endswith(":9011"))


class CompanionTests(unittest.TestCase):
    def test_khala_refuses_an_incomplete_app(self):
        with tempfile.TemporaryDirectory() as folder, \
                self.assertRaisesRegex(RuntimeError, "incomplete"):
            khala_server.launch_official_ui(Path(folder), Path(folder), 9)

    def test_yue2_refuses_a_missing_studio_and_writes_its_config(self):
        with tempfile.TemporaryDirectory() as folder:
            args = type("Args", (), {
                "webui_root": Path(folder), "output_root": Path(folder), "ui_port": 9,
                "memory_budget_gib": 20.0, "model_path": Path("/m"),
                "vae_path": Path("/v"), "device": "mps"})()
            with self.assertRaisesRegex(RuntimeError, "not installed"):
                yue2_server.launch_studio(args, "http://x")
            config = yue2_server.studio_config(args).read_text()
        self.assertIn("device: mps", config)
        self.assertIn(f"python: {sys.executable}", config)

    def test_khala_readiness_reads_json(self):
        ready = khala_server._says("status", "idle")
        self.assertTrue(ready(io.BytesIO(json.dumps({"status": "idle"}).encode())))
        self.assertFalse(ready(io.BytesIO(b'{"status": "busy"}')))
        self.assertTrue(yue2_server._answers_json(io.BytesIO(b"{}")))


class Child:
    """A companion process that stops when asked, or never."""

    def __init__(self, stops=True, exited=None):
        self.stops = stops
        self.exited = exited
        self.calls = []

    def poll(self):
        return self.exited

    def terminate(self):
        self.calls.append("terminate")

    def wait(self, timeout):
        if not self.stops:
            raise subprocess.TimeoutExpired("child", timeout)

    def kill(self):
        self.calls.append("kill")


class HostingTests(unittest.TestCase):
    def test_a_companion_that_will_not_stop_is_killed(self):
        stubborn, polite = Child(stops=False), Child()
        hosting.terminate(cast(Any, stubborn), 0.1)
        hosting.terminate(cast(Any, polite), 0.1)
        self.assertEqual((stubborn.calls, polite.calls), (["terminate", "kill"], ["terminate"]))
        khala_server._stop_all([cast(Any, Child(stops=False))])

    def test_stopping_is_wired_to_exit_and_signals(self):
        registered: list = []
        signals: list = []
        with patch.object(hosting.atexit, "register", registered.append), \
                patch.object(hosting.signal, "signal", lambda number, handler: signals.append(handler)):
            hosting.stop_on_exit(lambda: None)
        self.assertEqual(len(registered), 1)
        with self.assertRaises(SystemExit):
            signals[0](15, None)

    def test_waiting_ends_when_the_companion_answers_or_exits(self):
        answer = io.BytesIO(b"ok")
        with patch.object(hosting.urllib.request, "urlopen", lambda *a, **k: _Opened(answer)):
            hosting.wait_until_ready(hosting.Readiness("http://x", 1.0), "thing")
        with self.assertRaisesRegex(RuntimeError, "thing exited during startup"):
            hosting.wait_until_ready(hosting.Readiness("http://x", 1.0), "thing",
                                     cast(Any, Child(exited=1)))
        with patch.object(hosting.urllib.request, "urlopen", _refused), \
                patch.object(hosting, "POLL_S", 0.01), \
                self.assertRaisesRegex(RuntimeError, "did not become ready at http://x"):
            hosting.wait_until_ready(hosting.Readiness("http://x", 0.05), "thing")


class _Opened:
    def __init__(self, body):
        self.body = body

    def __enter__(self):
        return self.body

    def __exit__(self, *_):
        return False


def _refused(*_args, **_kwargs):
    raise OSError("refused")
