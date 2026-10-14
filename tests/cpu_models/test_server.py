"""Starting and stopping one processor model, with a stand-in process and clock."""

import subprocess
import tempfile
import unittest
from pathlib import Path

from ai_lab.cpu_models import CpuModel
from ai_lab.cpu_models.server import CpuServer, ProcessTools, command

MODEL = CpuModel(name="bge-m3", path="/m/bge.gguf", port=8390, idle_stop_s=60.0)


class FakeProcess:
    def __init__(self, exit_code=None, stubborn=False):
        self.exit_code = exit_code
        self.stubborn = stubborn
        self.terminated = self.killed = False

    def poll(self):
        return self.exit_code

    def terminate(self):
        self.terminated = True
        if not self.stubborn:
            self.exit_code = 0

    def kill(self):
        self.killed = True
        self.exit_code = -9

    def wait(self, timeout=None):
        if self.exit_code is None:
            raise subprocess.TimeoutExpired("llama-server", timeout)
        return self.exit_code


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class CpuServerTests(unittest.TestCase):
    def setUp(self):
        self.folder = Path(tempfile.mkdtemp())
        self.clock = Clock()
        self.launched = []
        self.process = FakeProcess()
        self.healthy_after = 0

    def launcher(self, argv, env, log):
        self.launched.append((argv, env, log))
        return self.process

    def probe(self, port):
        self.healthy_after -= 1
        return self.healthy_after < 0

    def server(self, model=MODEL):
        tools = ProcessTools(launch=self.launcher, clock=self.clock, sleep=self.clock.sleep)
        return CpuServer("/bin/llama-server", model, self.folder, self.probe, tools)

    def test_the_command_keeps_the_card_out_of_it(self):
        argv = command("/bin/llama-server", MODEL)
        self.assertIn("--embedding", argv)
        self.assertEqual(argv[argv.index("-ngl") + 1], "0")
        self.assertEqual(argv[argv.index("--pooling") + 1], "cls")
        self.assertEqual(argv[argv.index("--port") + 1], "8390")

    def test_started_once_and_waited_for(self):
        self.healthy_after = 3
        server = self.server()
        server.ensure_started()
        server.ensure_started()
        self.assertEqual(len(self.launched), 1)
        self.assertEqual(self.launched[0][1]["CUDA_VISIBLE_DEVICES"], "")
        self.assertTrue(server.running)

    def test_a_server_that_dies_while_starting_is_reported(self):
        self.healthy_after = 10
        self.process = FakeProcess(exit_code=1)
        with self.assertRaisesRegex(RuntimeError, "stopped while starting"):
            self.server().ensure_started()

    def test_a_server_that_never_answers_is_stopped(self):
        self.healthy_after = 10**9
        server = self.server()
        with self.assertRaisesRegex(TimeoutError, "did not start"):
            server.ensure_started()
        self.assertTrue(self.process.terminated)
        self.assertFalse(server.running)

    def test_idle_for_its_limit_it_stops(self):
        server = self.server()
        server.ensure_started()
        self.clock.now += 30
        self.assertFalse(server.stop_if_idle(busy=False))
        self.clock.now += 31
        self.assertFalse(server.stop_if_idle(busy=True))
        self.assertTrue(server.stop_if_idle(busy=False))
        self.assertFalse(server.running)

    def test_zero_keeps_it_running(self):
        server = self.server(CpuModel(name="m", path="/m", port=8391, idle_stop_s=0))
        server.ensure_started()
        self.clock.now += 10**6
        self.assertFalse(server.stop_if_idle(busy=False))

    def test_a_server_that_ignores_terminate_is_killed(self):
        self.process = FakeProcess(stubborn=True)
        server = self.server()
        server.ensure_started()
        server.stop()
        self.assertTrue(self.process.killed)
        server.stop()

    def test_the_real_launcher_logs_to_the_folder(self):
        server = CpuServer("/bin/sleep", MODEL, self.folder, lambda _port: True)
        server.ensure_started()
        server.stop()
        self.assertTrue((self.folder / "cpu-bge-m3.log").exists())


if __name__ == "__main__":
    unittest.main()
