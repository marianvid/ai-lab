import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from ai_lab.builds import Builds, SourceBuild, Version
from ai_lab.events import EventBus
from ai_lab.types import LogEvent


class VersionTests(unittest.TestCase):
    def test_a_build_tag_compares_numerically(self):
        """b10331 is older than b10433, which string comparison gets right by
        luck and would get wrong at four digits to five."""
        self.assertLess(Version("b10331").number, Version("b10433").number)
        self.assertLess(Version("b9999").number, Version("b10000").number)

    def test_an_unreadable_tag_is_zero_rather_than_an_error(self):
        self.assertEqual(Version("").number, 0)
        self.assertEqual(Version("unknown").number, 0)


class SourceBuildTests(unittest.TestCase):
    def setUp(self):
        self._temporary = TemporaryDirectory()
        self.path = Path(self._temporary.name)
        self.addCleanup(self._temporary.cleanup)
        (self.path / ".git").mkdir()
        self.bus = EventBus()
        self.build = SourceBuild("llamacpp", str(self.path), self.bus)

    def test_a_directory_without_git_is_reported_not_crashed(self):
        build = SourceBuild("llamacpp", "/nowhere", self.bus)
        self.assertFalse(build.exists)
        self.assertEqual(build.status()["installed"], "")
        with self.assertRaises(ValueError):
            build.check()

    def test_the_installed_version_comes_from_the_checkout(self):
        with patch.object(SourceBuild, "_git", side_effect=["b10331\n", "7ba604f1c\n"]):
            version = self.build.installed()
        self.assertEqual(version.tag, "b10331")
        self.assertEqual(version.commit, "7ba604f1c")

    def test_the_newest_remote_tag_wins(self):
        listing = "\n".join([
            "aaa\trefs/tags/b10331",
            "bbb\trefs/tags/b10433",
            "ccc\trefs/tags/b9999",
            "ddd\trefs/tags/master-something",
        ])
        with patch.object(SourceBuild, "_git",
                          side_effect=[listing, "b10331\n", "7ba604f1c\n"]):
            status = self.build.check()
        self.assertEqual(status["latest"], "b10433")
        self.assertTrue(status["update_available"])

    def test_no_update_offered_when_already_current(self):
        with patch.object(SourceBuild, "_git",
                          side_effect=["aaa\trefs/tags/b10331", "b10331\n", "7ba604f1c\n"]):
            status = self.build.check()
        self.assertFalse(status["update_available"])

    def test_a_remote_without_build_tags_is_an_error_not_a_silent_zero(self):
        with patch.object(SourceBuild, "_git", return_value=""):
            with self.assertRaises(ValueError):
                self.build.check()

    def test_output_is_streamed_line_by_line_and_kept(self):
        """A ten-minute compile with nothing on screen looks like a hang."""
        subscription = self.bus.subscribe()
        commands = []

        def fake_stream(build, command, elapsed, timeout):
            commands.append(command)
            build._say("out", f"ran {command[0]}", 0)

        with patch.object(SourceBuild, "_stream", fake_stream), \
             patch.object(SourceBuild, "_git", side_effect=["b10433\n", "abc1234\n"] * 4):
            self.build.update()
            self.build._thread.join(timeout=5)

        self.assertEqual(self.build.status()["state"], "done")
        self.assertEqual([item[0] for item in commands], ["git", "git", "cmake"])
        published = []
        stream = subscription.events(timeout=0.01)
        while (event := next(stream)) is not None:
            # The stream also carries change notices, which have no text.
            if isinstance(event, LogEvent):
                published.append(event.text)
        self.assertTrue(any("cmake" in item for item in published))

    def test_it_rebuilds_without_reconfiguring(self):
        """The existing build directory holds this machine's compile flags."""
        commands = []
        with patch.object(SourceBuild, "_stream",
                          lambda self, command, elapsed, timeout: commands.append(command)), \
             patch.object(SourceBuild, "_git", side_effect=["b1\n", "a\n"] * 4):
            self.build.update()
            self.build._thread.join(timeout=5)
        cmake = [item for item in commands if item[0] == "cmake"][0]
        self.assertEqual(cmake[:3], ["cmake", "--build", "build"])
        self.assertNotIn("-DGGML_CUDA=ON", cmake)

    def test_a_failure_is_recorded_with_its_message(self):
        def explode(self, command, elapsed, timeout):
            raise RuntimeError("cmake exited 2")

        with patch.object(SourceBuild, "_stream", explode), \
             patch.object(SourceBuild, "_git", side_effect=["b1\n", "a\n"] * 4):
            self.build.update()
            self.build._thread.join(timeout=5)
        status = self.build.status()
        self.assertEqual(status["state"], "failed")
        self.assertIn("cmake exited 2", status["error"])

    def test_two_builds_cannot_run_at_once(self):
        with patch.object(SourceBuild, "_stream",
                          lambda *a, **k: __import__("time").sleep(0.3)), \
             patch.object(SourceBuild, "_git", side_effect=["b1\n", "a\n"] * 4):
            self.build.update()
            with self.assertRaises(ValueError):
                self.build.update()
            self.build._thread.join(timeout=5)


class BuildsTests(unittest.TestCase):
    def test_only_engines_with_a_source_path_are_included(self):
        builds = Builds({"llamacpp": {"source": {"path": "/opt/ai/llama.cpp"}},
                         "vllm": {"binary": "/usr/bin/vllm"}}, EventBus())
        self.assertEqual([item["engine"] for item in builds.all()], ["llamacpp"])
        with self.assertRaises(KeyError):
            builds.get("vllm")

    def test_no_configuration_means_no_builds(self):
        self.assertEqual(Builds({}, EventBus()).all(), [])


if __name__ == "__main__":
    unittest.main()


class RobustnessTests(unittest.TestCase):
    """The settings screen reads this on every draw, so it must not throw."""

    def setUp(self):
        self._temporary = TemporaryDirectory()
        self.path = Path(self._temporary.name)
        self.addCleanup(self._temporary.cleanup)
        (self.path / ".git").mkdir()
        self.build = SourceBuild("llamacpp", str(self.path), EventBus())

    def test_a_checkout_git_cannot_read_shows_as_unknown(self):
        status = self.build.status()          # a .git directory that is not a repo
        self.assertEqual(status["installed"], "")
        self.assertTrue(status["exists"])

    def test_status_survives_a_repository_with_no_tags(self):
        with patch.object(SourceBuild, "_git", side_effect=ValueError("no tags")):
            self.assertEqual(self.build.installed().tag, "")


class UnknownVersionTests(unittest.TestCase):
    """A shallow clone has no tags, and that must not become a false offer."""

    def setUp(self):
        self._temporary = TemporaryDirectory()
        self.path = Path(self._temporary.name)
        self.addCleanup(self._temporary.cleanup)
        (self.path / ".git").mkdir()
        self.build = SourceBuild("llamacpp", str(self.path), EventBus())

    def test_no_update_is_offered_when_the_local_version_is_unknown(self):
        listing = "aaa\trefs/tags/b10433"
        with patch.object(SourceBuild, "_git", side_effect=[listing, ValueError("no tags")]):
            status = self.build.check()
        self.assertEqual(status["installed"], "")
        self.assertEqual(status["latest"], "b10433")
        self.assertFalse(status["update_available"])

    def test_the_reason_is_explained_rather_than_left_blank(self):
        self.assertIn("shallow", self.build.status()["note"])

    def test_a_missing_checkout_says_so(self):
        build = SourceBuild("llamacpp", "/nowhere", EventBus())
        self.assertIn("No git checkout", build.status()["note"])

    def test_a_healthy_checkout_has_nothing_to_report(self):
        with patch.object(SourceBuild, "_git", side_effect=["b10331\n", "abc\n"]):
            self.assertEqual(self.build.status()["note"], "")


class WatchTests(unittest.TestCase):
    """Versions are checked on a timer, so the screen is right when opened."""

    def setUp(self):
        self._temporary = TemporaryDirectory()
        self.path = Path(self._temporary.name)
        self.addCleanup(self._temporary.cleanup)
        (self.path / ".git").mkdir()

    def builds(self):
        return Builds({"llamacpp": {"source": {"path": str(self.path)}}}, EventBus())

    def test_it_checks_every_build_on_the_timer(self):
        import ai_lab.builds as module
        builds = self.builds()
        self.addCleanup(builds.stop)
        checked = []
        with patch.object(SourceBuild, "check", lambda self: checked.append(self.engine_id)), \
             patch.object(module, "STARTUP_DELAY_S", 0.01):
            builds.watch(interval_s=0.05)
            deadline = __import__("time").monotonic() + 3
            while not checked and __import__("time").monotonic() < deadline:
                __import__("time").sleep(0.01)
        self.assertIn("llamacpp", checked)

    def test_a_failure_is_swallowed_rather_than_shouted_about(self):
        """Being offline is not worth a message on a page nobody is reading."""
        import ai_lab.builds as module
        builds = self.builds()
        self.addCleanup(builds.stop)
        with patch.object(SourceBuild, "check", side_effect=ValueError("no network")), \
             patch.object(module, "STARTUP_DELAY_S", 0.01):
            builds.watch(interval_s=0.05)
            __import__("time").sleep(0.2)
        self.assertEqual(builds.all()[0]["state"], "idle")

    def test_watching_twice_starts_one_timer(self):
        builds = self.builds()
        self.addCleanup(builds.stop)
        builds.watch(interval_s=60)
        first = builds._timer
        builds.watch(interval_s=60)
        self.assertIs(builds._timer, first)

    def test_nothing_is_started_when_no_source_is_configured(self):
        builds = Builds({}, EventBus())
        builds.watch(interval_s=60)
        self.assertIsNone(builds._timer)
