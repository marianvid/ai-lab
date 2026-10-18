import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from ai_lab.builds.build_commands import BuildTools
from ai_lab.builds.core import Builds
from ai_lab.builds.source_build import BuildOptions, SourceBuild
from ai_lab.builds.source_versions import Version, mark_version
from ai_lab.events import EventBus
from ai_lab.types import LogEvent


def finish(build):
    """Wait for the build started in the background."""
    assert build.job.thread is not None
    build.job.thread.join(timeout=5)


class FakeGit:
    """git as these tests see it: answers in turn (like a mock's side_effect), every command recorded.

    `ancestor` is git's answer to "is that tag already in this history?":
    False means there is something new, True nothing, None that git cannot tell.
    """

    def __init__(self, side_effect=None, return_value="", ancestor=None):
        self.calls = Mock(side_effect=side_effect, return_value=return_value)
        self.ancestor = ancestor

    def run(self, path, *arguments, timeout=15):
        return self.calls(*arguments, timeout=timeout)

    def is_ancestor(self, path, tag, against):
        return self.ancestor


class FakeStream:
    """The long commands of an update: recorded, and `behaviour(command)` run for each."""

    def __init__(self, behaviour=None, says=None):
        self.commands = []
        self.behaviour = behaviour
        self.says = says

    def run(self, command, cwd, on_line, timeout):
        self.commands.append(command)
        if self.says:
            on_line(self.says(command))
        if self.behaviour:
            self.behaviour(command)


def build_with(path, bus, git=None, stream=None, options=None):
    """A SourceBuild whose git and long commands are the fakes given."""
    return SourceBuild("llamacpp", str(path), bus, options,
                       BuildTools(git=git, commands=stream))


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

    def build(self, git=None, stream=None, options=None):
        return build_with(self.path, self.bus, git, stream, options)

    def test_a_directory_without_git_is_reported_not_crashed(self):
        build = SourceBuild("llamacpp", "/nowhere", self.bus)
        self.assertFalse(build.exists)
        self.assertEqual(build.status()["installed"], "")
        with self.assertRaises(ValueError):
            build.check()

    def test_the_installed_version_comes_from_the_checkout(self):
        version = self.build(FakeGit(["b10331\n", "7ba604f1c\n"])).installed()
        self.assertEqual(version.tag, "b10331")
        self.assertEqual(version.commit, "7ba604f1c")

    def test_the_newest_tag_on_the_followed_line_wins(self):
        # check() fetches, lists the tags on its line, then reads the installed
        # version. git already sorts them, so the first match is the newest.
        git = FakeGit(["", "v0.2.0\nv0.1.10\nv0.1.2\n", "v0.1.2\n", "7ba604f1c\n"],
                      ancestor=False)
        status = self.build(git).check()
        self.assertEqual(status["latest"], "v0.2.0")
        self.assertEqual(status["line"], "stable")
        self.assertTrue(status["update_available"])

    def test_the_nightly_line_is_followed_when_that_is_what_was_asked_for(self):
        git = FakeGit(["", "b10587\nb10448\n", "b10448\n", "abc\n"], ancestor=False)
        status = self.build(git, options=BuildOptions(line="nightly")).check()
        self.assertEqual(status["latest"], "b10587")
        self.assertEqual(status["line"], "nightly")
        # It asked for b tags, not v tags.
        listing = [call for call in git.calls.call_args_list
                   if "for-each-ref" in call.args][0]
        self.assertIn("refs/tags/b*", listing.args)

    def test_an_unrecognised_line_falls_back_rather_than_breaking(self):
        build = SourceBuild("llamacpp", str(self.path), self.bus, BuildOptions(line="whatever"))
        self.assertEqual(build.line, "stable")

    def test_tags_from_the_other_line_are_ignored(self):
        # `git for-each-ref refs/tags/v*` would also match a tag called
        # "vendor-something". Only a real version is taken.
        git = FakeGit(["", "vendor-thing\nv0.1.2\n", "v0.1.2\n", "abc\n"], ancestor=True)
        self.assertEqual(self.build(git).check()["latest"], "v0.1.2")

    def test_no_update_offered_when_the_tag_is_already_in_this_history(self):
        # The exact question, and the one that works across both lines:
        # somebody on b10448 moving to v0.2.0 is going to a *smaller* number
        # and a newer thing, so the numbers cannot decide this.
        git = FakeGit(["", "v0.2.0\n", "v0.2.0\n", "abc\n"], ancestor=True)
        self.assertFalse(self.build(git).check()["update_available"])

    def test_moving_from_a_nightly_to_a_stable_tag_is_offered(self):
        git = FakeGit(["", "v0.2.0\n", "b10448\n", "abc\n"], ancestor=False)
        status = self.build(git).check()
        self.assertEqual(status["installed"], "b10448")
        self.assertEqual(status["latest"], "v0.2.0")
        self.assertTrue(status["update_available"],
                        "a smaller number on the other line is still an update")

    def test_a_remote_without_tags_on_that_line_is_an_error_not_a_silent_zero(self):
        with self.assertRaises(ValueError):
            self.build(FakeGit(return_value="")).check()

    def test_output_is_streamed_line_by_line_and_kept(self):
        """A ten-minute compile with nothing on screen looks like a hang."""
        subscription = self.bus.subscribe()
        stream = FakeStream(says=lambda command: f"ran {command[0]}")
        build = self.build(FakeGit(["v0.2.0\n"] * 40), stream)
        build.update()
        finish(build)

        self.assertEqual(build.status()["state"], "done")
        commands = stream.commands
        self.assertEqual([item[0] for item in commands], ["git", "git", "cmake"])
        checkout = [item for item in commands if item[:2] == ["git", "checkout"]]
        self.assertTrue(checkout, "an update must move to a named tag")
        self.assertIn("ran cmake", build.status()["log"])
        published = []
        events = subscription.events(timeout=0.01)
        while (event := next(events)) is not None:
            # The stream also carries change notices, which have no text.
            if isinstance(event, LogEvent):
                published.append(event.text)
        self.assertTrue(any("cmake" in item for item in published))

    def test_it_rebuilds_without_reconfiguring(self):
        """The existing build directory holds this machine's compile flags."""
        stream = FakeStream()
        build = self.build(FakeGit(["v0.1.0\n"] * 40), stream)
        build.update()
        finish(build)
        cmake = [item for item in stream.commands if item[0] == "cmake"][0]
        self.assertEqual(cmake[:3], ["cmake", "--build", "build"])
        self.assertNotIn("-DGGML_CUDA=ON", cmake)

    def test_a_failure_is_recorded_with_its_message(self):
        def explode(command):
            raise RuntimeError("cmake exited 2")

        build = self.build(FakeGit(["v0.1.0\n"] * 40), FakeStream(explode))
        build.update()
        finish(build)
        status = build.status()
        self.assertEqual(status["state"], "failed")
        self.assertIn("cmake exited 2", status["error"])

    def test_two_builds_cannot_run_at_once(self):
        started, release = threading.Event(), threading.Event()

        def hold(command):
            started.set()
            release.wait(5)

        build = self.build(FakeGit(["v0.1.0\n"] * 40), FakeStream(hold))
        build.update()
        self.assertTrue(started.wait(5))
        with self.assertRaises(ValueError):
            build.update()
        release.set()
        finish(build)

    def test_a_version_is_not_switched_or_removed_while_a_build_runs(self):
        started, release = threading.Event(), threading.Event()

        def hold(command):
            started.set()
            release.wait(5)

        build = self.build(FakeGit(["v0.1.0\n"] * 40), FakeStream(hold))
        build.update()
        self.assertTrue(started.wait(5))
        try:
            with self.assertRaisesRegex(ValueError, "while .* running"):
                build.activate("b10000-abc")
            with self.assertRaisesRegex(ValueError, "while .* running"):
                build.remove("b10000-abc")
        finally:
            release.set()
            finish(build)


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
        mark_version(self.legacy, Version("b10448", "oldcommit"))
        self.builds = self.root / "compiled"
        self.builds.mkdir()
        self.git = FakeGit(return_value="")
        self.stream = FakeStream()
        self.build = self.made(self.git, self.stream)
        self.build.point_at(self.legacy)

    def made(self, git, stream):
        return build_with(self.source, EventBus(), git, stream, BuildOptions(
            builds=str(self.builds), legacy_build=str(self.legacy),
            cmake_args=["-DGGML_CUDA=ON", "-DCMAKE_CUDA_ARCHITECTURES=native"]))

    def _compiled(self, tag="v0.3.0", commit="newcommit"):
        path = self.builds / f"build-{tag}"
        (path / "bin").mkdir(parents=True)
        (path / "bin" / "llama-server").write_text("binary")
        mark_version(path, Version(tag, commit))
        return path

    def test_the_active_binary_not_checkout_is_the_installed_version(self):
        # The checkout says v0.3.0; the binary that runs is the legacy b10448.
        build = self.made(FakeGit(["v0.3.0\n", "newcommit\n"] * 4), FakeStream())
        self.assertEqual(build.checkout_version(), Version("v0.3.0", "newcommit"))
        self.assertEqual(build.installed(), Version("b10448", "oldcommit"))

    def test_compiled_versions_are_listed_and_the_active_one_is_marked(self):
        newer = self._compiled()
        found = {item.version.tag: item.active for item in self.build.environments()}
        self.assertEqual(found, {"b10448": True, "v0.3.0": False})
        self.assertTrue(newer.exists())

    def test_switching_is_instant_and_deletes_nothing(self):
        newer = self._compiled()
        self.build.activate(newer.name)
        self.assertEqual(self.build.active(), newer.resolve())
        self.assertTrue(self.legacy.exists())
        self.assertIn("newcommit", self.git.calls.call_args.args)

    def test_the_active_build_cannot_be_deleted(self):
        with self.assertRaises(ValueError):
            self.build.remove(self.legacy.name)
        self.assertTrue(self.legacy.exists())

    def test_an_inactive_build_can_be_deleted(self):
        newer = self._compiled()
        self.build.remove(newer.name)
        self.assertFalse(newer.exists())

    def test_update_configures_and_builds_beside_the_active_one(self):
        def make_folder(command):
            if command[:2] == ["cmake", "-S"]:
                target = Path(command[command.index("-B") + 1])
                (target / "bin").mkdir(parents=True)
                (target / "bin" / "llama-server").write_text("binary")

        stream = FakeStream(make_folder)
        build = self.made(FakeGit(["v0.3.0\n", "v0.3.0\n", "newcommit\n"]), stream)
        build.update()
        finish(build)
        commands = stream.commands

        status = build.status()
        self.assertEqual(status["state"], "done", status["error"])
        self.assertEqual(status["installed"], "v0.3.0")
        self.assertEqual({item["version"] for item in status["environments"]},
                         {"b10448", "v0.3.0"})
        configure = next(command for command in commands if command[:2] == ["cmake", "-S"])
        self.assertIn("-DGGML_CUDA=ON", configure)
        self.assertIn("-DCMAKE_CUDA_ARCHITECTURES=native", configure)
        self.assertTrue(any(command[-1:] == ["--version"] for command in commands))

    def test_a_failed_new_build_leaves_the_active_one_and_no_partial_copy(self):
        def fail_compiling(command):
            if command[:3] == ["cmake", "--build", str(self.builds / "build-v0.3.0")]:
                raise RuntimeError("compile failed")
            if command[:2] == ["cmake", "-S"]:
                Path(command[command.index("-B") + 1]).mkdir(parents=True)

        build = self.made(FakeGit(["v0.3.0\n", "v0.3.0\n", "newcommit\n", ""]),
                          FakeStream(fail_compiling))
        build.update()
        finish(build)

        self.assertEqual(build.status()["state"], "failed")
        self.assertEqual(build.active(), self.legacy.resolve())
        self.assertFalse((self.builds / "build-v0.3.0").exists())


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
        build = build_with(self.path, EventBus(), FakeGit(side_effect=ValueError("no tags")))
        self.assertEqual(build.installed().tag, "")


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
        git = FakeGit(["", "v0.2.0\n", ValueError("no tags")])
        status = build_with(self.path, EventBus(), git).check()
        self.assertEqual(status["installed"], "")
        self.assertEqual(status["latest"], "v0.2.0")
        self.assertFalse(status["update_available"])

    def test_the_reason_is_explained_rather_than_left_blank(self):
        self.assertIn("shallow", self.build.status()["note"])

    def test_a_missing_checkout_says_so(self):
        build = SourceBuild("llamacpp", "/nowhere", EventBus())
        self.assertIn("No git checkout", build.status()["note"])

    def test_a_healthy_checkout_has_nothing_to_report(self):
        build = build_with(self.path, EventBus(), FakeGit(["b10331\n", "abc\n"]))
        self.assertEqual(build.status()["note"], "")


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
        import ai_lab.builds.core as module
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
        import ai_lab.builds.core as module
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
        first = builds.watcher.thread
        builds.watch(interval_s=60)
        self.assertIs(builds.watcher.thread, first)

    def test_nothing_is_started_when_no_source_is_configured(self):
        builds = Builds({}, EventBus())
        builds.watch(interval_s=60)
        self.assertIsNone(builds.watcher.thread)


if __name__ == "__main__":
    unittest.main()
