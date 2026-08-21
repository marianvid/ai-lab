import unittest
from pathlib import Path
from unittest.mock import patch

from ai_lab.hosts.command import Result
from ai_lab.hosts.linux import CONTROL_TIMEOUT_S, LinuxHost


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
            return result(show(qwen="active 4321"))

        with patch("ai_lab.hosts.linux.run", side_effect=fake_run):
            status = host.status("qwen")
        self.assertTrue(status.running)
        self.assertEqual(status.pid, 4321)
        self.assertIn("ai-lab-engine@qwen.service", seen[0])
        self.assertEqual(len(seen), 1, "one instance should still be one command")


def show(**instances):
    """systemd's answer to `systemctl show`, in its own shape.

    The properties come back in whatever order systemd likes — on the real
    machine MainPID arrives before Id — so the blocks here are written that way
    rather than tidily, and a parser that relies on position fails against them.
    """
    blocks = []
    for identifier, spec in instances.items():
        state, pid = spec.split()
        blocks.append(f"MainPID={pid}\n"
                      f"Id=ai-lab-engine@{identifier}.service\n"
                      f"ActiveState={state}")
    return "\n\n".join(blocks) + "\n"


class BatchStatusTests(unittest.TestCase):
    """Asking systemd about every instance at once.

    One at a time meant three commands per instance. Measured on the container
    with eleven instances that was 152 ms, and it was the entire cost of
    drawing the model list — the readiness probes beside it were free. The
    gateway asks the same question twice per request, so a workflow of short
    calls paid half a second to reach an engine answering in 17 ms.
    """

    def statuses(self, output, ids, ok=True):
        with patch("ai_lab.hosts.linux.run",
                   return_value=result(output, returncode=0 if ok else 1)) as call:
            found = LinuxHost().statuses(ids)
        self.assertEqual(call.call_count, 1, "the whole point is one command")
        return found

    def test_every_instance_comes_back_from_one_command(self):
        found = self.statuses(
            show(a="active 11", b="inactive 0",
                 c="failed 0"), ["a", "b", "c"])
        self.assertEqual([found[key].running for key in "abc"],
                         [True, False, False])
        self.assertEqual(found["a"].pid, 11)

    def test_blocks_are_matched_by_name_not_by_position(self):
        # systemd is under no obligation to answer in the order it was asked.
        found = self.statuses(show(b="active 22", a="inactive 0"),
                              ["a", "b"])
        self.assertTrue(found["b"].running)
        self.assertFalse(found["a"].running)
        self.assertEqual(found["b"].pid, 22)

    def test_a_stale_pid_on_a_stopped_unit_is_not_reported(self):
        # A pid belonging to a process that has exited would be sampled for
        # accelerator memory and answer with somebody else's numbers.
        found = self.statuses(show(a="inactive 999"), ["a"])
        self.assertIsNone(found["a"].pid)

    def test_an_instance_systemd_never_mentions_is_still_answered(self):
        found = self.statuses(show(a="active 11"), ["a", "ghost"])
        self.assertIn("ghost", found)
        self.assertFalse(found["ghost"].running)

    def test_a_failed_command_reports_stopped_rather_than_raising(self):
        # Every other reader here treats "could not ask" and "answered no" the
        # same way, because the alternative is a page that will not draw.
        found = self.statuses("", ["a", "b"], ok=False)
        self.assertEqual([found[key].running for key in "ab"], [False, False])

    def test_asking_about_nothing_runs_no_command(self):
        with patch("ai_lab.hosts.linux.run") as call:
            self.assertEqual(LinuxHost().statuses([]), {})
        call.assert_not_called()

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


class StopTimeoutTests(unittest.TestCase):
    """The wait has to outlast the thing being waited for.

    `systemctl stop` blocks until the unit is stopped, and the unit gives an
    engine `TimeoutStopSec` seconds to go quietly before systemd kills it. If
    this side's patience equals systemd's, the stop times out at the very
    moment it succeeds — which is what happened on the real machine: a vLLM
    instance stopped in the middle of generating really did stop, the card
    really was released, and the interface said "sudo: timed out after 60s".

    Two numbers in two files that must stay in a particular order is exactly
    the kind of pair that drifts apart silently, so it is asserted rather than
    left to a comment.
    """

    UNIT = (Path(__file__).resolve().parents[2]
            / "system" / "ai-lab-engine@.service")

    def unit_stop_seconds(self) -> float:
        for line in self.UNIT.read_text().splitlines():
            stripped = line.strip()
            if stripped.startswith("TimeoutStopSec="):
                return float(stripped.split("=", 1)[1].rstrip("s"))
        self.fail("the unit no longer sets TimeoutStopSec")

    def test_the_unit_still_bounds_how_long_a_stop_takes(self):
        # Without it, systemd waits 90 seconds by default and an engine that
        # ignores SIGTERM holds the card for all of them.
        self.assertLessEqual(self.unit_stop_seconds(), 60.0)

    def test_we_wait_longer_than_systemd_does(self):
        self.assertGreater(
            CONTROL_TIMEOUT_S, self.unit_stop_seconds(),
            "this side must outlast the unit, or a successful stop is "
            "reported as a timeout")

    def test_with_a_margin_rather_than_by_a_second(self):
        # Equal was the bug; one second more would be the same bug with better
        # luck. systemd still has to reap the processes after it kills them.
        self.assertGreaterEqual(CONTROL_TIMEOUT_S - self.unit_stop_seconds(), 30.0)
