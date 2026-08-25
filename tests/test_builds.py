import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from ai_lab.builds import Builds, SourceBuild, Version, _mark
from ai_lab.events import EventBus
from ai_lab.types import LogEvent


class VersionTests(unittest.TestCase):
    def test_a_build_tag_compares_numerically(self):
        """b10331 is older than b10433, which string comparison gets right by
        luck and would get wrong at four digits to five."""
        self.assertLess(Version("b10331").number, Version("b10433").number)
        self.assertLess(Version("b9999").number, Version("b10000").number)

    def test_a_stable_version_compares_part_by_part(self):
        """v0.1.10 is newer than v0.1.2, which string comparison gets wrong."""
        self.assertLess(Version("v0.1.2").number, Version("v0.1.10").number)
        self.assertLess(Version("v0.1.9").number, Version("v0.2.0").number)

    def test_an_unreadable_tag_has_nothing_to_compare_rather_than_erroring(self):
        self.assertEqual(Version("").number, ())
        self.assertEqual(Version("unknown").number, ())

    def test_a_tag_says_which_line_it_is_on(self):
        # The two lines are different things: b tags are made on nearly every
        # commit to master, v tags are chosen releases. Telling them apart is
        # what stops the numbers being compared across them.
        self.assertEqual(Version("b10448").line, "nightly")
        self.assertEqual(Version("v0.2.0").line, "stable")
        self.assertEqual(Version("").line, "")
        self.assertEqual(Version("master").line, "")


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

    def test_the_newest_tag_on_the_followed_line_wins(self):
        # check() fetches, lists the tags on its line, then reads the installed
        # version. git already sorts them, so the first match is the newest.
        with patch.object(SourceBuild, "_git",
                          side_effect=["", "v0.2.0\nv0.1.10\nv0.1.2\n",
                                       "v0.1.2\n", "7ba604f1c\n"]), \
             patch.object(SourceBuild, "_is_ahead", return_value=True):
            status = self.build.check()
        self.assertEqual(status["latest"], "v0.2.0")
        self.assertEqual(status["line"], "stable")
        self.assertTrue(status["update_available"])

    def test_the_nightly_line_is_followed_when_that_is_what_was_asked_for(self):
        build = SourceBuild("llamacpp", str(self.path), self.bus, line="nightly")
        with patch.object(SourceBuild, "_git",
                          side_effect=["", "b10587\nb10448\n",
                                       "b10448\n", "abc\n"]) as git, \
             patch.object(SourceBuild, "_is_ahead", return_value=True):
            status = build.check()
        self.assertEqual(status["latest"], "b10587")
        self.assertEqual(status["line"], "nightly")
        # It asked for b tags, not v tags.
        listing = [call for call in git.call_args_list
                   if "for-each-ref" in call.args][0]
        self.assertIn("refs/tags/b*", listing.args)

    def test_an_unrecognised_line_falls_back_rather_than_breaking(self):
        build = SourceBuild("llamacpp", str(self.path), self.bus, line="whatever")
        self.assertEqual(build.line, "stable")

    def test_tags_from_the_other_line_are_ignored(self):
        # `git for-each-ref refs/tags/v*` would also match a tag called
        # "vendor-something". Only a real version is taken.
        with patch.object(SourceBuild, "_git",
                          side_effect=["", "vendor-thing\nv0.1.2\n",
                                       "v0.1.2\n", "abc\n"]), \
             patch.object(SourceBuild, "_is_ahead", return_value=False):
            status = self.build.check()
        self.assertEqual(status["latest"], "v0.1.2")

    def test_no_update_offered_when_the_tag_is_already_in_this_history(self):
        # The exact question, and the one that works across both lines:
        # somebody on b10448 moving to v0.2.0 is going to a *smaller* number
        # and a newer thing, so the numbers cannot decide this.
        with patch.object(SourceBuild, "_git",
                          side_effect=["", "v0.2.0\n", "v0.2.0\n", "abc\n"]), \
             patch.object(SourceBuild, "_is_ahead", return_value=False):
            status = self.build.check()
        self.assertFalse(status["update_available"])

    def test_moving_from_a_nightly_to_a_stable_tag_is_offered(self):
        with patch.object(SourceBuild, "_git",
                          side_effect=["", "v0.2.0\n", "b10448\n", "abc\n"]), \
             patch.object(SourceBuild, "_is_ahead", return_value=True):
            status = self.build.check()
        self.assertEqual(status["installed"], "b10448")
        self.assertEqual(status["latest"], "v0.2.0")
        self.assertTrue(status["update_available"],
                        "a smaller number on the other line is still an update")

    def test_a_remote_without_tags_on_that_line_is_an_error_not_a_silent_zero(self):
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
             patch.object(SourceBuild, "_git", side_effect=["v0.2.0\n"] * 40):
            self.build.update()
            self.build._thread.join(timeout=5)

        self.assertEqual(self.build.status()["state"], "done")
        self.assertEqual([item[0] for item in commands], ["git", "git", "cmake"])
        checkout = [item for item in commands if item[:2] == ["git", "checkout"]]
        self.assertTrue(checkout, "an update must move to a named tag")
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
             patch.object(SourceBuild, "_git", side_effect=["v0.1.0\n"] * 40):
            self.build.update()
            self.build._thread.join(timeout=5)
        cmake = [item for item in commands if item[0] == "cmake"][0]
        self.assertEqual(cmake[:3], ["cmake", "--build", "build"])
        self.assertNotIn("-DGGML_CUDA=ON", cmake)

    def test_a_failure_is_recorded_with_its_message(self):
        def explode(self, command, elapsed, timeout):
            raise RuntimeError("cmake exited 2")

        with patch.object(SourceBuild, "_stream", explode), \
             patch.object(SourceBuild, "_git", side_effect=["v0.1.0\n"] * 40):
            self.build.update()
            self.build._thread.join(timeout=5)
        status = self.build.status()
        self.assertEqual(status["state"], "failed")
        self.assertIn("cmake exited 2", status["error"])

    def test_two_builds_cannot_run_at_once(self):
        with patch.object(SourceBuild, "_stream",
                          lambda *a, **k: __import__("time").sleep(0.3)), \
             patch.object(SourceBuild, "_git", side_effect=["v0.1.0\n"] * 40):
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


class VersionedBuildTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        (self.source / ".git").mkdir(parents=True)
        self.legacy = self.source / "build"
        (self.legacy / "bin").mkdir(parents=True)
        (self.legacy / "bin" / "llama-server").write_text("binary")
        _mark(self.legacy, Version("b10448", "oldcommit"))
        self.builds = self.root / "compiled"
        self.builds.mkdir()
        self.build = SourceBuild(
            "llamacpp", str(self.source), EventBus(), builds=str(self.builds),
            legacy_build=str(self.legacy),
            cmake_args=["-DGGML_CUDA=ON", "-DCMAKE_CUDA_ARCHITECTURES=native"])
        self.build.point_at(self.legacy)

    def _compiled(self, tag="v0.3.0", commit="newcommit"):
        path = self.builds / f"build-{tag}"
        (path / "bin").mkdir(parents=True)
        (path / "bin" / "llama-server").write_text("binary")
        _mark(path, Version(tag, commit))
        return path

    def test_the_active_binary_not_checkout_is_the_installed_version(self):
        with patch.object(SourceBuild, "checkout_version",
                          return_value=Version("v0.3.0", "newcommit")):
            self.assertEqual(self.build.installed(), Version("b10448", "oldcommit"))

    def test_compiled_versions_are_listed_and_the_active_one_is_marked(self):
        newer = self._compiled()
        found = {item.version.tag: item.active for item in self.build.environments()}
        self.assertEqual(found, {"b10448": True, "v0.3.0": False})
        self.assertTrue(newer.exists())

    def test_switching_is_instant_and_deletes_nothing(self):
        newer = self._compiled()
        with patch.object(SourceBuild, "_git", return_value="") as git:
            self.build.activate(newer.name)
        self.assertEqual(self.build.active(), newer.resolve())
        self.assertTrue(self.legacy.exists())
        self.assertIn("newcommit", git.call_args.args)

    def test_the_active_build_cannot_be_deleted(self):
        with self.assertRaises(ValueError):
            self.build.remove(self.legacy.name)
        self.assertTrue(self.legacy.exists())

    def test_an_inactive_build_can_be_deleted(self):
        newer = self._compiled()
        self.build.remove(newer.name)
        self.assertFalse(newer.exists())

    def test_update_configures_and_builds_beside_the_active_one(self):
        commands = []

        def stream(owner, command, elapsed, timeout):
            commands.append(command)
            if command[:2] == ["cmake", "-S"]:
                target = Path(command[command.index("-B") + 1])
                (target / "bin").mkdir(parents=True)
                (target / "bin" / "llama-server").write_text("binary")

        with patch.object(SourceBuild, "_stream", stream), \
             patch.object(SourceBuild, "_git",
                          side_effect=["v0.3.0\n", "v0.3.0\n", "newcommit\n"]):
            self.build.update()
            self.build._thread.join(timeout=5)

        status = self.build.status()
        self.assertEqual(status["state"], "done", status["error"])
        self.assertEqual(status["installed"], "v0.3.0")
        self.assertEqual({item["version"] for item in status["environments"]},
                         {"b10448", "v0.3.0"})
        configure = next(command for command in commands if command[:2] == ["cmake", "-S"])
        self.assertIn("-DGGML_CUDA=ON", configure)
        self.assertIn("-DCMAKE_CUDA_ARCHITECTURES=native", configure)
        self.assertTrue(any(command[-1:] == ["--version"] for command in commands))

    def test_a_failed_new_build_leaves_the_active_one_and_no_partial_copy(self):
        def stream(owner, command, elapsed, timeout):
            if command[:3] == ["cmake", "--build", str(self.builds / "build-v0.3.0")]:
                raise RuntimeError("compile failed")
            if command[:2] == ["cmake", "-S"]:
                Path(command[command.index("-B") + 1]).mkdir(parents=True)

        with patch.object(SourceBuild, "_stream", stream), \
             patch.object(SourceBuild, "_git",
                          side_effect=["v0.3.0\n", "v0.3.0\n", "newcommit\n", ""]):
            self.build.update()
            self.build._thread.join(timeout=5)

        self.assertEqual(self.build.status()["state"], "failed")
        self.assertEqual(self.build.active(), self.legacy.resolve())
        self.assertFalse((self.builds / "build-v0.3.0").exists())


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
        # A shallow clone has no tags, so there is nothing to compare against
        # and nothing sensible to move from. Offering an update there would be
        # offering to replace an unknown with an unknown.
        with patch.object(SourceBuild, "_git",
                          side_effect=["", "v0.2.0\n", ValueError("no tags")]):
            status = self.build.check()
        self.assertEqual(status["installed"], "")
        self.assertEqual(status["latest"], "v0.2.0")
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
