"""scripts/deploy.sh's failure paths, run for real against a stand-in ssh.

The reviewer's item 17: the ERR trap that did not fire inside `remote()`
(7246e7d) would have passed the whole suite. Here the script runs in a copy of
the project with one trivial test, and every far-side command goes to a
stand-in `ssh` that records it and fails where it is told to.

**A real ssh is refused:** the stand-in is first on PATH and checked before
every run, the host is under `.invalid` (a name that never resolves), and the
key does not exist; a real ssh reached by mistake has nowhere to go
(ai-lab's mistake of 2026-11-27 11:24 UTC, an empty folder on the Proxmox host).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "deploy.sh"
HOST = "root@ssh-stand-in.invalid"
STAND_IN = r'''#!/usr/bin/env bash
# The far side, recorded and never reached.
cat > /dev/null
command="${@: -1}"
echo "$command" >> "$(dirname "$0")/commands.log"
case "$command" in
  *"unittest discover"*) [ -n "$FAIL_TESTS" ] && { echo "far-side tests failed" >&2; exit 1; } ;;
  *"pip"*) [ -n "$FAIL_PIP" ] && { echo "pip: no space left" >&2; exit 1; } ;;
  *ai_lab.hosts.ready*) [ -n "$FAIL_READY" ] && { echo "the manager stopped answering" >&2; exit 1; } ;;
  *NRestarts*) echo 0 ;;
esac
exit 0
'''
RESTORE_MARK = "mv /opt/ai-lab.prev/$item /opt/ai-lab/"


@unittest.skipUnless(shutil.which("bash") and shutil.which("tar"), "needs bash and tar")
class DeployScriptTests(unittest.TestCase):
    def setUp(self):
        self.place = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.place, ignore_errors=True)
        project = self.place / "project"
        (project / "scripts").mkdir(parents=True)
        shutil.copy2(SCRIPT, project / "scripts" / "deploy.sh")
        comfy = project / "scripts" / "install-comfyui-linux.sh"
        comfy.write_text("#!/usr/bin/env bash\nexit 0\n")
        comfy.chmod(0o755)
        (project / "tests").mkdir()
        (project / "tests" / "__init__.py").write_text("")
        (project / "tests" / "test_one.py").write_text(
            "import unittest\n\nclass One(unittest.TestCase):\n"
            "    def test_one(self):\n        self.assertTrue(True)\n")
        self.script = project / "scripts" / "deploy.sh"
        self.bin = self.place / "bin"
        self.bin.mkdir()
        ssh = self.bin / "ssh"
        ssh.write_text(STAND_IN)
        ssh.chmod(0o755)
        (self.bin / "python3").symlink_to(sys.executable)
        self.log = self.bin / "commands.log"

    def deploy(self, **fail: str) -> subprocess.CompletedProcess:
        path = os.pathsep.join([str(self.bin), "/usr/bin", "/bin"])
        found = shutil.which("ssh", path=path)
        self.assertEqual(Path(found or "").resolve(), (self.bin / "ssh").resolve(),
                         "the stand-in ssh is not the one deploy.sh would run")
        env = {"PATH": path, "HOME": str(self.place), "AI_LAB_HOST": HOST,
               "AI_LAB_CTID": "999", "AI_LAB_SSH_KEY": str(self.place / "no-such-key"), **fail}
        return subprocess.run(["bash", str(self.script)], env=env, capture_output=True,
                              stdin=subprocess.DEVNULL, text=True, timeout=120, check=False)

    def commands(self) -> str:
        return self.log.read_text() if self.log.exists() else ""

    def test_a_clean_run_swaps_restarts_and_restores_nothing(self):
        done = self.deploy()
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn("Deployment completed", done.stdout)
        self.assertIn("systemctl restart ai-lab.service", self.commands())
        self.assertNotIn(RESTORE_MARK, self.commands())

    def test_a_far_side_failure_inside_remote_puts_the_previous_code_back(self):
        # pip fails in step 5, inside remote(): the ERR trap must fire there (set -E).
        done = self.deploy(FAIL_PIP="1")
        self.assertEqual(done.returncode, 1)
        self.assertIn("putting the previous code back", done.stderr)
        self.assertIn(RESTORE_MARK, self.commands())

    def test_a_manager_that_will_not_answer_is_rolled_back(self):
        done = self.deploy(FAIL_READY="1")
        self.assertEqual(done.returncode, 1)
        self.assertIn(RESTORE_MARK, self.commands())
        self.assertNotIn("Deployment completed", done.stdout)

    def test_failing_tests_in_the_container_stop_before_anything_is_swapped(self):
        done = self.deploy(FAIL_TESTS="1")
        self.assertNotEqual(done.returncode, 0)
        self.assertNotIn("/opt/ai-lab.prev", self.commands(), "the running code was touched")
        self.assertNotIn(RESTORE_MARK, self.commands())

    def test_every_far_side_command_goes_through_the_container(self):
        # A bare `a && b` would run b on the Proxmox host (2026-11-27 11:24 UTC).
        self.deploy()
        prefix = "pct exec '999' -- "
        for command in self.commands().splitlines():
            with self.subTest(command[:60]):
                self.assertTrue(command.startswith(prefix), command)
                inside = command[len(prefix):]
                if "&&" in inside or "||" in inside:
                    self.assertTrue(inside.startswith("sh -lc '"),
                                    f"a chained command outside one container shell: {inside[:80]}")


if __name__ == "__main__":
    unittest.main()
