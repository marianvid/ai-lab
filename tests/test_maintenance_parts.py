"""The small shared pieces behind builds, installs and storage, each on its own."""

import subprocess
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from ai_lab.changes import fromgit
from ai_lab.disk import free_bytes, tree_size
from ai_lab.events import EventBus
from ai_lab.builds.install_log import Command, InstallLog, since
from ai_lab.builds.package_environments import replace_link
from ai_lab.builds.public_services import address
from ai_lab.types import LogEvent
from ai_lab.builds.update_watch import UpdateWatcher


class DiskTests(unittest.TestCase):
    def test_a_folder_counts_its_files_and_a_file_itself(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a").write_bytes(b"x" * 10)
            (root / "sub").mkdir()
            (root / "sub" / "b").write_bytes(b"y" * 5)
            (root / "link").symlink_to(root / "a")
            self.assertEqual(tree_size(root), 15)
            self.assertEqual(tree_size(root / "a"), 10)
            self.assertEqual(tree_size(root / "gone"), 0)

    def test_free_space_is_asked_of_the_nearest_existing_folder(self):
        with TemporaryDirectory() as directory:
            self.assertGreater(free_bytes(Path(directory) / "not" / "yet"), 0)
        self.assertEqual(free_bytes(None), 0)


class Item:
    def __init__(self, fails=False):
        self.checked = 0
        self.fails = fails

    def check(self):
        self.checked += 1
        if self.fails:
            raise OSError("offline")


class WatcherTests(unittest.TestCase):
    def test_every_item_is_checked_and_a_failure_is_ignored(self):
        items = [Item(fails=True), Item()]
        watcher = UpdateWatcher(lambda: items, "test-watch")
        self.addCleanup(watcher.stop)
        watcher.start(interval_s=60, startup_delay_s=0.01)
        deadline = time.monotonic() + 3
        while items[1].checked == 0 and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual([item.checked for item in items], [1, 1])

    def test_nothing_to_check_starts_nothing_and_twice_starts_once(self):
        empty = UpdateWatcher(list, "empty")
        empty.start(60, 60)
        self.assertIsNone(empty.thread)
        watcher = UpdateWatcher(lambda: [Item()], "once")
        self.addCleanup(watcher.stop)
        watcher.start(60, 60)
        first = watcher.thread
        watcher.start(60, 60)
        self.assertIs(watcher.thread, first)


class InstallLogTests(unittest.TestCase):
    def setUp(self):
        self.bus = EventBus()
        self.log = InstallLog("vllm", self.bus)

    def test_output_is_kept_and_a_failure_names_the_command(self):
        with TemporaryDirectory() as directory:
            command = Command(["sh", "-c", "echo one; echo two; exit 3"], Path(directory),
                              {"PATH": "/bin:/usr/bin"}, 30)
            with self.assertRaisesRegex(ValueError, "sh -c echo one; echo two; exit 3 failed"):
                self.log.run(command, since(time.monotonic()))
        self.assertEqual(list(self.log.lines), ["one", "two"])

    def test_a_line_is_published(self):
        seen: list[LogEvent] = []
        with mock.patch.object(self.bus, "publish", seen.append):
            self.log.say("status", "hello", 5)
        self.assertIsInstance(seen[0], LogEvent)
        self.assertEqual(seen[0].text, "hello")


class LinkTests(unittest.TestCase):
    def test_the_link_is_replaced_in_one_step(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "one").mkdir()
            (root / "two").mkdir()
            link = root / "current"
            replace_link(link, root / "one")
            replace_link(link, root / "two")
            self.assertEqual(link.resolve(), (root / "two").resolve())
            self.assertFalse((root / ".current-new").exists())


class PublicServicesTests(unittest.TestCase):
    def test_every_address_is_a_template_for_https(self):
        for name in ("pypi_package", "huggingface_api", "huggingface_file", "github_releases"):
            self.assertTrue(address(name).startswith("https://"), name)

    def test_an_unknown_name_says_where_it_looked(self):
        with self.assertRaisesRegex(KeyError, "public_services.json"):
            address("nowhere")


class FromGitTests(unittest.TestCase):
    def test_commits_waiting_are_read_newest_first_with_their_area(self):
        answer = subprocess.CompletedProcess([], 0, "abc123\x1fcuda: faster kernels\n"
                                                    "def456\x1fdocs: typo\n", "")
        with mock.patch.object(fromgit.subprocess, "run", return_value=answer):
            changes, cut = fromgit.waiting("/nowhere", fetch=False)
        self.assertEqual([item.reference for item in changes], ["abc123", "def456"])
        self.assertEqual(cut, 0)

    def test_git_s_own_words_come_back_as_a_value_error(self):
        refused = subprocess.CompletedProcess([], 128, "", "fatal: not a git repository\n")
        with mock.patch.object(fromgit.subprocess, "run", return_value=refused), \
                self.assertRaisesRegex(ValueError, "not a git repository"):
            fromgit.waiting("/nowhere")

    def test_more_than_the_most_are_cut_and_counted(self):
        lines = "".join(f"h{n}\x1ftitle {n}\n" for n in range(fromgit.MOST + 5))
        answer = subprocess.CompletedProcess([], 0, lines, "")
        with mock.patch.object(fromgit.subprocess, "run", return_value=answer):
            changes, cut = fromgit.waiting("/nowhere", fetch=False)
        self.assertEqual((len(changes), cut), (fromgit.MOST, 5))


if __name__ == "__main__":
    unittest.main()
