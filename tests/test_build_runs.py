"""The long commands of builds and installs, run for real on harmless programs or recorded."""

import json
import os
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

from ai_lab.builds.build_commands import SubprocessStream
from ai_lab.builds.gitapps import GitApplicationInstall, GitCommands, git_output, remote_commit
from ai_lab.builds.install_log import Command, InstallLog
from ai_lab.config.policy import BuildLimits
from ai_lab.events import EventBus


class Folder(unittest.TestCase):
    def setUp(self):
        self._folder = tempfile.TemporaryDirectory()
        self.addCleanup(self._folder.cleanup)
        self.folder = Path(self._folder.name)


class CommandStreamTests(Folder):
    def test_each_line_is_handed_on_as_it_comes(self):
        lines: list[str] = []
        SubprocessStream().run(["sh", "-c", "echo one; echo two >&2"], self.folder,
                               lines.append, 10)
        self.assertEqual(sorted(lines), ["one", "two"])

    def test_a_failing_command_names_its_exit_code(self):
        with self.assertRaisesRegex(RuntimeError, "failed with exit code 3"):
            SubprocessStream().run(["sh", "-c", "exit 3"], self.folder, lambda _line: None, 10)

    def test_a_command_past_its_time_is_killed(self):
        with self.assertRaisesRegex(TimeoutError, r"sh exceeded 0.2s"):
            SubprocessStream().run(["sh", "-c", "echo a; sleep 0.4; echo b; sleep 5"],
                                   self.folder, lambda _line: None, 0.2)


class SilentCommandTests(Folder):
    """A command that hangs without printing is stopped by its limit, and nothing is left behind."""

    def silent(self):
        pid_file = self.folder / "pid"
        return ["sh", "-c", f"echo $$ > {pid_file}; exec sleep 30"], pid_file

    def assert_gone(self, pid_file):
        pid = int(pid_file.read_text())
        with self.assertRaises(ProcessLookupError):   # not even a zombie: it was waited for
            os.kill(pid, 0)

    def test_a_silent_build_step_is_stopped_at_its_limit(self):
        argv, pid_file = self.silent()
        started = time.monotonic()
        with self.assertRaisesRegex(TimeoutError, "sh exceeded 0.5s"):
            SubprocessStream().run(argv, self.folder, lambda _line: None, 0.5)
        self.assertLess(time.monotonic() - started, 3)
        self.assert_gone(pid_file)

    def test_a_silent_step_with_children_is_stopped_whole(self):
        """A build is a tree: the compiler under `make` is what hangs, and it holds the pipe."""
        child_pid = self.folder / "child"
        argv = ["sh", "-c", f"sleep 30 & echo $! > {child_pid}; sleep 30"]
        started = time.monotonic()
        with self.assertRaisesRegex(TimeoutError, "sh exceeded 0.5s"):
            SubprocessStream().run(argv, self.folder, lambda _line: None, 0.5)
        self.assertLess(time.monotonic() - started, 3)
        for _ in range(50):                 # the child is reaped by init, a moment later
            try:
                os.kill(int(child_pid.read_text()), 0)
            except ProcessLookupError:
                break
            time.sleep(0.05)
        self.assert_gone(child_pid)

    def test_a_command_that_closes_its_output_and_runs_on_is_stopped_too(self):
        started = time.monotonic()
        with self.assertRaisesRegex(TimeoutError, "exceeded 0.5s"):
            SubprocessStream().run(["sh", "-c", "exec >/dev/null 2>&1; sleep 30"],
                                   self.folder, lambda _line: None, 0.5)
        self.assertLess(time.monotonic() - started, 3)

    def test_a_silent_install_step_is_stopped_at_its_limit(self):
        argv, pid_file = self.silent()
        started = time.monotonic()
        with self.assertRaisesRegex(ValueError, "took longer than 0.5s"):
            InstallLog("comfyui", None).run(
                Command(argv, self.folder, {"PATH": "/usr/bin:/bin"}, 0.5), lambda: 0)
        self.assertLess(time.monotonic() - started, 3)
        self.assert_gone(pid_file)


class InstallLogTests(Folder):
    def test_a_command_s_lines_reach_the_log_and_a_failure_is_said(self):
        log = InstallLog("comfyui", None)
        log.run(Command(["sh", "-c", "echo fetched"], self.folder, {"PATH": "/usr/bin:/bin"}, 10),
                lambda: 0)
        self.assertIn("fetched", log.kept())
        with self.assertRaisesRegex(ValueError, "took longer than 0.2s"):
            log.run(Command(["sh", "-c", "echo a; sleep 0.4; echo b; sleep 5"], self.folder,
                            {"PATH": "/usr/bin:/bin"}, 0.2), lambda: 0)
        with self.assertRaisesRegex(ValueError, "failed"):
            log.run(Command(["sh", "-c", "exit 1"], self.folder, {"PATH": "/usr/bin:/bin"}, 10),
                    lambda: 0)


class GitQuestionsTests(Folder):
    def git(self, *args):
        subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
                       cwd=self.folder, check=True, capture_output=True)

    def test_a_checkout_and_its_remote_are_asked_with_the_machines_path(self):
        self.git("init", "-q")
        (self.folder / "file").write_text("x")
        self.git("add", "file")
        self.git("commit", "-q", "-m", "first")
        head = git_output(self.folder, "rev-parse", "HEAD").strip()
        self.assertEqual(len(head), 40)
        self.assertEqual(remote_commit(str(self.folder), "HEAD",
                                       system_path="/usr/local/bin:/usr/bin:/bin"), head)

    def test_a_question_git_cannot_answer_is_a_readable_refusal(self):
        with self.assertRaises(ValueError):
            git_output(self.folder / "nowhere", "status")
        with self.assertRaises(ValueError):
            remote_commit(str(self.folder / "nowhere"), "HEAD")


class GitApplicationBuildTests(Folder):
    """A ComfyUI release built from start to finish, every command recorded instead of run."""

    def test_a_release_is_cloned_given_its_environment_checked_and_switched_to(self):
        ran = []

        def run(_log, command, _elapsed):
            ran.append(command)
            if command.argv[:2] == ["git", "clone"]:
                app = Path(command.argv[-1])
                (app / "custom_nodes" / "extra").mkdir(parents=True)
                (app / "requirements.txt").write_text("torch\n")
                (app / "custom_nodes" / "extra" / "requirements.txt").write_text("numpy\n")
            if command.argv[1:2] == ["venv"]:
                python = Path(command.argv[2]) / "bin" / "python"
                python.parent.mkdir(parents=True)
                python.write_text("")
        install = GitApplicationInstall(
            "comfyui", {"source": {"root": str(self.folder), "repository": "upstream",
                                   "uv": "uv", "python": "python3.12"}},
            EventBus(), GitCommands(remote_commit=lambda *_args, **_kw: "c" * 40, run=run),
            BuildLimits(install_s=1234, system_path="/opt/bin"))
        install.install()
        assert install.job.thread is not None
        install.job.thread.join(timeout=5)
        self.assertEqual(install.job.state, "done", install.job.error)
        steps = [command.argv[:3] for command in ran]
        self.assertEqual(steps[0], ["git", "clone", "--no-checkout"])
        installs = [command.argv[-1] for command in ran if command.argv[1:3] == ["pip", "install"]]
        self.assertEqual([Path(path).parent.name for path in installs], ["ComfyUI", "extra"])
        check = ran[-1]
        self.assertIn("importlib.import_module('server')", check.argv[-1])
        self.assertEqual({command.environment["PATH"] for command in ran}, {"/opt/bin"})
        self.assertEqual({command.timeout_s for command in ran[:-1]}, {1234})
        release = install.active()
        assert release is not None
        meta = json.loads((release / ".ai-lab-release.json").read_text())
        self.assertEqual(meta["core_commit"], "c" * 40)


if __name__ == "__main__":
    unittest.main()
