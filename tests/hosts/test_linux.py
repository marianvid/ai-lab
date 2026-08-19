import unittest
from unittest.mock import patch

from ai_lab.hosts.command import Result
from ai_lab.hosts.linux import LinuxHost


def result(stdout="", returncode=0, stderr=""):
    return Result(returncode, stdout, stderr)


class AcceleratorParsingTests(unittest.TestCase):
    """nvidia-smi output is parsed here rather than on a machine with a GPU."""

    def parse(self, line):
        return LinuxHost._parse(line)

    def test_a_normal_reading(self):
        snapshot = self.parse("NVIDIA RTX PRO 4500 Blackwell, 24817, 32623, 41, 7")
        self.assertTrue(snapshot.available)
        self.assertEqual(snapshot.name, "NVIDIA RTX PRO 4500 Blackwell")
        self.assertEqual(snapshot.memory_used_mb, 24817)
        self.assertEqual(snapshot.memory_total_mb, 32623)
        self.assertEqual(snapshot.temperature_c, 41)
        self.assertEqual(snapshot.memory_kind, "dedicated")

    def test_unsupported_fields_do_not_break_the_reading(self):
        """Some fields report [N/A] on some cards."""
        snapshot = self.parse("Tesla T4, 100, 16000, [N/A], [N/A]")
        self.assertTrue(snapshot.available)
        self.assertEqual(snapshot.memory_used_mb, 100)
        self.assertIsNone(snapshot.temperature_c)

    def test_a_truncated_line_is_reported_as_no_accelerator(self):
        self.assertFalse(self.parse("garbage").available)


class ControlTests(unittest.TestCase):
    def test_starting_writes_the_command_then_calls_the_helper(self):
        host = LinuxHost(control_helper="/usr/local/sbin/ai-lab-control")
        from ai_lab.types import ProcessSpec

        with patch("ai_lab.hosts.linux.launch.write_spec") as write, \
             patch("ai_lab.hosts.linux.run", return_value=result()) as run:
            host.start(ProcessSpec("qwen", ["llama-server", "--model", "x"], {}))
        write.assert_called_once()
        argv = run.call_args[0][0]
        self.assertEqual(argv[:4],
                         ["sudo", "-n", "/usr/local/sbin/ai-lab-control", "start"])
        self.assertEqual(argv[4], "qwen")

    def test_a_failing_helper_raises_with_its_message(self):
        host = LinuxHost()
        with patch("ai_lab.hosts.linux.run",
                   return_value=result(returncode=1, stderr="unit not found")):
            with self.assertRaises(RuntimeError) as caught:
                host.stop("qwen")
        self.assertIn("unit not found", str(caught.exception))

    def test_status_reads_the_templated_unit(self):
        host = LinuxHost()
        seen = []

        def fake_run(argv, timeout=10.0):
            seen.append(argv)
            if "is-active" in argv:
                return result("active\n")
            if "is-enabled" in argv:
                return result("enabled\n")
            return result("4321\n")

        with patch("ai_lab.hosts.linux.run", side_effect=fake_run):
            status = host.status("qwen")
        self.assertTrue(status.running)
        self.assertTrue(status.enabled)
        self.assertEqual(status.pid, 4321)
        self.assertIn("ai-lab-engine@qwen.service", seen[0])

    def test_no_nvidia_smi_means_no_accelerator_not_a_crash(self):
        with patch("ai_lab.hosts.linux.run",
                   return_value=result(returncode=127, stderr="not found")):
            snapshot = LinuxHost().accelerator()
        self.assertFalse(snapshot.available)
        self.assertEqual(snapshot.kind, "none")


if __name__ == "__main__":
    unittest.main()


class EngineDetectionTests(unittest.TestCase):
    """vLLM is installed in a virtual environment, so PATH does not see it."""

    def capabilities(self, host, on_path=()):
        with patch("ai_lab.hosts.linux.which",
                   side_effect=lambda name: f"/usr/bin/{name}" if name in on_path else None), \
             patch.object(LinuxHost, "accelerator",
                          return_value=type("S", (), {"kind": "cuda"})()):
            return host.capabilities()

    def test_a_configured_path_counts_as_installed(self):
        host = LinuxHost(vllm_binary="/opt/ai/vllm/.venv/bin/vllm")
        self.assertIn("vllm", self.capabilities(host).engines)

    def test_without_a_path_or_a_binary_it_is_absent(self):
        host = LinuxHost()
        self.assertNotIn("vllm", self.capabilities(host).engines)

    def test_path_alone_still_counts(self):
        host = LinuxHost()
        self.assertIn("vllm", self.capabilities(host, on_path=("vllm",)).engines)

    def test_llama_cpp_is_unaffected(self):
        host = LinuxHost(vllm_binary="/opt/ai/vllm/.venv/bin/vllm")
        engines = self.capabilities(host, on_path=("llama-server",)).engines
        self.assertIn("llamacpp", engines)
