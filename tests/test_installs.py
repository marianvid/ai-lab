"""Installing a package engine beside the one that works.

The rule every test here defends: what is working is never written to. A new
version goes in a new folder, is checked that it starts, and only then does the
engine begin using it — so going back is always possible.

That is not theoretical. The vLLM installed on the container when this was
written could not be reinstalled: its wheel had left the local cache and the
index it came from was recorded nowhere.
"""

from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from ai_lab.events import EventBus
from ai_lab.installs import CURRENT, Installs, PackageInstall


def _environment(root: Path, version: str) -> Path:
    """A folder shaped like an installed environment, without 8 GB in it."""
    path = root / f".venv-{version}"
    (path / "bin").mkdir(parents=True)
    (path / "bin" / "python").write_text("#!/bin/sh\n")
    (path / "lib").mkdir()
    (path / "lib" / "big").write_bytes(b"x" * 1000)
    return path


class WhatIsInstalled(unittest.TestCase):
    def setUp(self):
        self._temporary = TemporaryDirectory()
        self.root = Path(self._temporary.name)
        self.addCleanup(self._temporary.cleanup)
        self.install = PackageInstall("vllm", str(self.root), "vllm", EventBus())

    def test_it_lists_the_folders_and_marks_the_one_in_use(self):
        old = _environment(self.root, "0.26.1")
        new = _environment(self.root, "0.27.1")
        self.install.point_at(new)

        found = {item.version: item.active for item in self.install.environments()}
        self.assertEqual(found, {"0.26.1": False, "0.27.1": True})
        self.assertEqual(self.install.active(), new.resolve())
        self.assertTrue(old.is_dir(), "pointing elsewhere deleted the old one")

    def test_it_ignores_anything_that_is_not_one_of_ours(self):
        _environment(self.root, "0.27.1")
        (self.root / "notes.txt").write_text("hello")
        (self.root / "some-other-folder").mkdir()
        self.assertEqual([item.version for item in self.install.environments()],
                         ["0.27.1"])

    def test_a_directory_that_does_not_exist_is_empty_not_an_error(self):
        install = PackageInstall("vllm", "/nowhere/at/all", "vllm", EventBus())
        self.assertEqual(install.environments(), [])
        self.assertIsNone(install.active())

    def test_it_reports_how_much_the_spare_copies_cost(self):
        _environment(self.root, "0.26.1")
        new = _environment(self.root, "0.27.1")
        self.install.point_at(new)
        status = self.install.status()
        # Only the one not in use is spare — the other is not a saving
        # available to anybody. One folder holds 1,000 bytes of "weights" and
        # a ten-byte stub for python.
        self.assertEqual(status["spare_bytes"], 1010)
        self.assertTrue(status["free_bytes"] > 0)


class Switching(unittest.TestCase):
    def setUp(self):
        self._temporary = TemporaryDirectory()
        self.root = Path(self._temporary.name)
        self.addCleanup(self._temporary.cleanup)
        self.install = PackageInstall("vllm", str(self.root), "vllm", EventBus())
        self.old = _environment(self.root, "0.26.1")
        self.new = _environment(self.root, "0.27.1")

    def test_going_back_is_the_same_act_as_going_forward(self):
        self.install.point_at(self.new)
        self.install.activate(".venv-0.26.1")
        self.assertEqual(self.install.active(), self.old.resolve())
        # And back again, because that is the point of keeping both.
        self.install.activate(".venv-0.27.1")
        self.assertEqual(self.install.active(), self.new.resolve())

    def test_the_link_never_points_at_nothing(self):
        # Swapping by deleting and recreating leaves a moment where anything
        # starting fails for a reason nobody would guess. The replacement is
        # done in one step instead, so the link is always valid.
        self.install.point_at(self.old)
        seen = []
        real = Path.replace

        def watch(self_path, target):
            seen.append((self.root / CURRENT).resolve(strict=False).name)
            return real(self_path, target)

        with patch.object(Path, "replace", watch):
            self.install.point_at(self.new)
        self.assertEqual(seen, [self.old.name],
                         "the link was not valid right up to the swap")
        self.assertEqual(self.install.active(), self.new.resolve())

    def test_an_unknown_name_is_refused_by_name(self):
        with self.assertRaises(KeyError):
            self.install.activate(".venv-does-not-exist")

    def test_a_folder_without_a_bin_is_refused(self):
        broken = self.root / ".venv-broken"
        broken.mkdir()
        with self.assertRaises(ValueError):
            self.install.point_at(broken)


class Removing(unittest.TestCase):
    def setUp(self):
        self._temporary = TemporaryDirectory()
        self.root = Path(self._temporary.name)
        self.addCleanup(self._temporary.cleanup)
        self.install = PackageInstall("vllm", str(self.root), "vllm", EventBus())
        self.old = _environment(self.root, "0.26.1")
        self.new = _environment(self.root, "0.27.1")
        self.install.point_at(self.new)

    def test_the_old_one_goes_when_asked(self):
        self.install.remove(".venv-0.26.1")
        self.assertFalse(self.old.exists())
        self.assertEqual(self.install.active(), self.new.resolve())

    def test_the_one_in_use_is_refused(self):
        # Otherwise the engine is left with nothing to run and the only sign
        # is a load that fails with a missing file.
        with self.assertRaises(ValueError) as refusal:
            self.install.remove(".venv-0.27.1")
        self.assertIn("in use", str(refusal.exception))
        self.assertTrue(self.new.exists())

    def test_nothing_is_ever_removed_on_its_own(self):
        # The previous version is the way back. There is no rule anywhere that
        # decides it has stopped being needed.
        self.install.install = lambda *a, **k: None      # not what is being tested
        for item in (self.install.environments(), self.install.status()):
            pass
        self.assertTrue(self.old.exists(), "something tidied it away")


class Installing(unittest.TestCase):
    """The install itself, with uv replaced by something that leaves folders."""

    def setUp(self):
        self._temporary = TemporaryDirectory()
        self.root = Path(self._temporary.name)
        self.addCleanup(self._temporary.cleanup)
        self.working = _environment(self.root, "0.26.1")
        (self.working / "lib" / "proof").write_text("the one that works")
        self.install = PackageInstall("vllm", str(self.root), "vllm", EventBus())
        self.install.point_at(self.working)

    def _pretend(self, lands="0.27.1", starts=True, fails_at=None):
        """Stand in for uv and for the check that the result starts."""
        def stream(command, elapsed, timeout, venv=None):
            if fails_at and fails_at in command[1]:
                raise ValueError(f"{fails_at} failed")
            if command[1] == "venv":
                built = Path(command[2])
                (built / "bin").mkdir(parents=True, exist_ok=True)
                (built / "bin" / "python").write_text("#!/bin/sh\n")

        def verify(environment, elapsed):
            if not starts:
                raise ValueError("The new environment does not start: no CUDA")

        return (patch.object(PackageInstall, "_stream",
                             lambda self, *a, **k: stream(*a, **k)),
                patch.object(PackageInstall, "_verify",
                             lambda self, *a, **k: verify(*a, **k)),
                patch("ai_lab.installs._version_in", lambda path, package: lands))

    def _run(self, **how):
        one, two, three = self._pretend(**how)
        with one, two, three:
            self.install.install(how.get("lands", "0.27.1"))
            self.install._thread.join(timeout=10)
        return self.install.status()

    def test_a_good_install_lands_beside_the_old_one_and_takes_over(self):
        status = self._run()
        self.assertEqual(status["state"], "done", status["error"])
        versions = {item["version"]: item["active"] for item in status["environments"]}
        self.assertEqual(versions, {"0.26.1": False, "0.27.1": True})
        # The proof that nothing was written over what worked.
        self.assertEqual((self.working / "lib" / "proof").read_text(),
                         "the one that works")

    def test_an_environment_that_will_not_start_never_takes_over(self):
        status = self._run(starts=False)
        self.assertEqual(status["state"], "failed")
        self.assertIn("does not start", status["error"])
        self.assertEqual(self.install.active(), self.working.resolve(),
                         "the engine was pointed at something that does not run")
        self.assertEqual([item["version"] for item in status["environments"]],
                         ["0.26.1"], "a broken environment was left lying about")

    def test_a_failed_download_leaves_nothing_behind_and_changes_nothing(self):
        status = self._run(fails_at="pip")
        self.assertEqual(status["state"], "failed")
        self.assertEqual(self.install.active(), self.working.resolve())
        self.assertFalse((self.root / ".venv-installing").exists())

    def test_installing_a_version_that_is_already_here_is_refused(self):
        status = self._run(lands="0.26.1")
        self.assertEqual(status["state"], "failed")
        self.assertIn("already installed", status["error"])
        self.assertEqual((self.working / "lib" / "proof").read_text(),
                         "the one that works")

    def test_two_installs_cannot_run_at_once(self):
        # Held open deliberately. Asking a second time "while the first is
        # still going" is only a test if the first is definitely still going,
        # and a version of this that relied on the first being slow passed on
        # one machine and failed on the other.
        import threading
        holding, started = threading.Event(), threading.Event()

        def slow(self_install, command, elapsed, timeout, venv=None):
            if command[1] == "venv":
                built = Path(command[2])
                (built / "bin").mkdir(parents=True, exist_ok=True)
                (built / "bin" / "python").write_text("#!/bin/sh\n")
            started.set()
            holding.wait(timeout=10)

        with patch.object(PackageInstall, "_stream", slow), \
             patch.object(PackageInstall, "_verify", lambda *a, **k: None), \
             patch("ai_lab.installs._version_in", lambda path, package: "0.27.1"):
            self.install.install("0.27.1")
            self.assertTrue(started.wait(timeout=5), "the first never started")
            with self.assertRaises(ValueError):
                self.install.install("0.27.1")
            holding.set()
            self.install._thread.join(timeout=10)
        self.assertEqual(self.install.status()["state"], "done")

    def test_configured_minimum_versions_are_checked_before_activation(self):
        install = PackageInstall(
            "paddleocr", str(self.root), "paddleocr", EventBus(),
            minimum_versions={"paddlepaddle-gpu": "3.3.0"})
        commands = []

        def run(command, **kwargs):
            commands.append(command)
            return type("Result", (), {"returncode": 0, "stdout": "ok\n",
                                        "stderr": ""})()

        with patch("ai_lab.installs.subprocess.run", run):
            install._verify(self.working, lambda: 0)
        program = commands[0][2]
        self.assertIn("paddlepaddle-gpu", program)
        self.assertIn("3.3.0", program)


class WhichEnginesHaveOne(unittest.TestCase):
    def test_only_an_engine_configured_as_packages_gets_one(self):
        installs = Installs({
            "llamacpp": {"binary": "/opt/ai/llama.cpp/build/bin/llama-server",
                         "source": {"path": "/opt/ai/llama.cpp"}},
            "vllm": {"binary": "/opt/ai/vllm/current/bin/vllm",
                     "source": {"package": "vllm"}},
        }, EventBus())
        self.assertIn("vllm", installs)
        self.assertNotIn("llamacpp", installs)
        with self.assertRaises(KeyError):
            installs.get("llamacpp")

    def test_package_version_requirements_are_wired_from_configuration(self):
        installs = Installs({
            "paddleocr": {
                "binary": "/opt/ai/paddleocr/current/bin/python",
                "source": {"package": "paddleocr",
                           "minimum_versions": {"paddlepaddle-gpu": "3.3.0"}},
            }}, EventBus())
        self.assertEqual(installs.get("paddleocr").minimum_versions,
                         {"paddlepaddle-gpu": "3.3.0"})

    def test_the_folder_is_worked_out_from_the_launch_path(self):
        installs = Installs({"vllm": {"binary": "/opt/ai/vllm/current/bin/vllm",
                                      "source": {"package": "vllm"}}}, EventBus())
        self.assertEqual(str(installs.get("vllm").root), "/opt/ai/vllm")


if __name__ == "__main__":
    unittest.main()


class TheOneThatWasAlreadyThere(unittest.TestCase):
    """A plain `.venv`, from before any of this existed.

    It cannot be renamed into the scheme. A virtual environment's launcher
    scripts carry in their first line the absolute path they were built at, so
    `/opt/ai/vllm/.venv/bin/vllm` starts `/opt/ai/vllm/.venv/bin/python` by
    name — move the folder and it starts nothing. Measured on the container,
    where a copy taken as a precaution had a first line pointing back at the
    original and would not have worked without it.
    """

    def setUp(self):
        self._temporary = TemporaryDirectory()
        self.root = Path(self._temporary.name)
        self.addCleanup(self._temporary.cleanup)
        self.first = self.root / ".venv"
        (self.first / "bin").mkdir(parents=True)
        (self.first / "bin" / "python").write_text("#!/bin/sh\n")
        self.install = PackageInstall("vllm", str(self.root), "vllm", EventBus())

    def test_it_is_listed_and_can_be_used(self):
        with patch("ai_lab.installs._version_in", lambda path, package: "0.26.1"):
            self.install.point_at(self.first)
            found = self.install.environments()
        self.assertEqual([item.name for item in found], [".venv"])
        self.assertEqual(found[0].version, "0.26.1",
                         "its version must be read, since its name has none")
        self.assertTrue(found[0].active)

    def test_it_is_marked_as_the_one_that_cannot_move(self):
        with patch("ai_lab.installs._version_in", lambda path, package: "0.26.1"):
            found = self.install.environments()[0]
        self.assertFalse(found.movable)

    def test_a_new_install_sits_beside_it_without_touching_it(self):
        (self.first / "bin" / "vllm").write_text("#!/tmp/x/.venv/bin/python\n")
        newer = self.root / ".venv-0.27.1"
        (newer / "bin").mkdir(parents=True)
        (newer / "bin" / "python").write_text("#!/bin/sh\n")
        self.install.point_at(newer)
        with patch("ai_lab.installs._version_in", lambda path, package: "0.26.1"):
            found = {item.name: item.active for item in self.install.environments()}
        self.assertEqual(found, {".venv": False, ".venv-0.27.1": True})
        self.assertTrue((self.first / "bin" / "vllm").exists())


class WhatIsWaiting(unittest.TestCase):
    """A package engine has to know what it could become without being asked.

    llama.cpp is asked with git on a timer, so its page is already right when
    it opens. vLLM had no equivalent and so had nothing to show until somebody
    pressed something — which is exactly the button this project removed. One
    request to the index answers it: 251 ms measured on the container, against
    2.2 s for a full package resolution.
    """

    def setUp(self):
        self._temporary = TemporaryDirectory()
        self.root = Path(self._temporary.name)
        self.addCleanup(self._temporary.cleanup)
        _environment(self.root, "0.26.1")
        self.install = PackageInstall("vllm", str(self.root), "vllm", EventBus())
        self.install.point_at(self.root / ".venv-0.26.1")

    def _index_says(self, version, fails=False):
        class Answer:
            def __enter__(inner):
                return inner

            def __exit__(inner, *args):
                return False

            def read(inner):
                return json.dumps({"info": {"version": version}}).encode()

        def open_url(url, timeout=None):
            if fails:
                raise OSError("no network")
            return Answer()

        return patch("ai_lab.installs.urllib.request.urlopen", open_url)

    def test_a_newer_version_upstream_is_an_update_waiting(self):
        with self._index_says("0.27.1"):
            status = self.install.check()
        self.assertEqual(status["installed"], "0.26.1")
        self.assertEqual(status["latest"], "0.27.1")
        self.assertTrue(status["update_available"])

    def test_the_same_version_is_not(self):
        with self._index_says("0.26.1"):
            status = self.install.check()
        self.assertFalse(status["update_available"])

    def test_an_index_that_cannot_be_reached_is_unknown_not_up_to_date(self):
        # An engine nobody could ask about must not look like one with nothing
        # waiting. Being offline is not good news.
        with self._index_says("", fails=True):
            status = self.install.check()
        self.assertEqual(status["latest"], "")
        self.assertFalse(status["update_available"],
                         "an unknown version must not be offered as an update")

    def test_nothing_installed_means_nothing_to_update(self):
        empty = PackageInstall("vllm", str(Path(self._temporary.name) / "gone"),
                               "vllm", EventBus())
        with self._index_says("0.27.1"):
            status = empty.check()
        self.assertEqual(status["installed"], "")
        self.assertFalse(status["update_available"])

    def test_the_version_in_use_is_the_one_reported(self):
        _environment(self.root, "0.27.1")
        self.install.point_at(self.root / ".venv-0.27.1")
        self.assertEqual(self.install.installed_now(), "0.27.1")
