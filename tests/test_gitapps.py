import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from ai_lab.events import EventBus
from ai_lab.gitapps import GitApplicationInstall


class GitApplicationInstallTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.install = GitApplicationInstall("comfyui", {
            "source": {"root": str(self.root), "kind": "git-app",
                       "repository": "https://example.invalid/ComfyUI.git",
                       "application": "ComfyUI"}}, EventBus())

    def release(self, name, commit):
        release = self.root / name
        (release / ".venv" / "bin").mkdir(parents=True)
        (release / ".venv" / "bin" / "python").write_text("")
        (release / "ComfyUI" / "custom_nodes").mkdir(parents=True)
        (release / ".ai-lab-release.json").write_text(json.dumps({
            "version": commit[:12], "core_commit": commit, "components": {}}))
        return release

    def test_switch_is_atomic_and_preserves_previous_release(self):
        first = self.release(".release-first", "a" * 40)
        second = self.release(".release-second", "b" * 40)
        self.install.point_at(first)
        self.install.point_at(second)
        self.assertEqual(self.install.active(), second.resolve())
        self.assertTrue(first.exists())

    def test_active_release_cannot_be_removed(self):
        active = self.release(".release-first", "a" * 40)
        self.install.point_at(active)
        with self.assertRaises(ValueError):
            self.install.remove(active.name)

    def test_failed_candidate_is_removed_without_moving_current(self):
        active = self.release(".release-first", "a" * 40)
        self.install.point_at(active)
        with patch.object(self.install, "_remote_commit", return_value="b" * 40), \
             patch.object(self.install, "_stream", side_effect=ValueError("install failed")):
            self.install.install("HEAD")
            self.install._thread.join(timeout=2)
        self.assertEqual(self.install.active(), active.resolve())
        self.assertEqual(self.install.status()["state"], "failed")
        self.assertFalse((self.root / (".release-" + "b" * 12)).exists())

    def test_dirty_custom_node_is_never_updated(self):
        active = self.release(".release-first", "a" * 40)
        node = active / "ComfyUI" / "custom_nodes" / "example"
        (node / ".git").mkdir(parents=True)
        self.install.point_at(active)
        with patch.object(self.install, "_git", return_value=" M changed.py\n"):
            with self.assertRaises(ValueError):
                self.install.update_component("example")


if __name__ == "__main__":
    unittest.main()
