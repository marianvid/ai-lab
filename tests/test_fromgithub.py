import unittest
from unittest import mock

from ai_lab.changes import fromgithub

RELEASES = [  # newest first, as GitHub returns them
    {"tag_name": "v0.4.0", "name": "0.4.0", "body": "four"},
    {"tag_name": "v0.3.0", "name": "", "body": ""},
    {"tag_name": "v0.2.0", "name": "0.2.0", "body": "two"},
    {"tag_name": "nightly", "name": "nightly", "body": "no version"},
    {"tag_name": "b10448", "name": "b10448", "body": "builds"},
]


def between(installed, latest, releases=RELEASES):
    with mock.patch.object(fromgithub, "_releases", return_value=releases):
        return fromgithub.between("owner/repo", installed, latest)


class BetweenTests(unittest.TestCase):
    def test_notes_after_installed_up_to_latest(self):
        notes, warning = between("v0.2.0", "v0.3.0")
        self.assertEqual(notes, "# v0.3.0\n\n_No notes were written for this release._")
        self.assertEqual(warning, "")

    def test_several_releases_are_joined_newest_first(self):
        notes, _ = between("0.2.0", "0.4.0")
        self.assertTrue(notes.startswith("# 0.4.0\n\nfour\n\n---\n\n# v0.3.0"))
        self.assertNotIn("two", notes)

    def test_different_tag_shapes_describe_only_the_target(self):
        notes, warning = between("b10448", "v0.4.0")
        self.assertEqual(notes, "# 0.4.0\n\nfour")
        self.assertIn("different lines", warning)

    def test_nothing_newer_says_so(self):
        notes, warning = between("v0.4.0", "v0.4.0")
        self.assertEqual(notes, "")
        self.assertIn("No release notes", warning)

    def test_an_unreachable_service_is_a_warning(self):
        with mock.patch.object(fromgithub, "_releases", side_effect=ValueError("down")):
            self.assertEqual(fromgithub.between("o/r", "v1", "v2"),
                             ("", "Could not read the release notes: down"))


if __name__ == "__main__":
    unittest.main()
