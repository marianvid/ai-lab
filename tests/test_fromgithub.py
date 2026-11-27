import unittest

from ai_lab.changes import fromgithub

RELEASES = [  # newest first, as GitHub returns them
    {"tag_name": "v0.4.0", "name": "0.4.0", "body": "four"},
    {"tag_name": "v0.3.0", "name": "", "body": ""},
    {"tag_name": "v0.2.0", "name": "0.2.0", "body": "two"},
    {"tag_name": "nightly", "name": "nightly", "body": "no version"},
    {"tag_name": "b10448", "name": "b10448", "body": "builds"},
]


def between(installed, latest, releases=RELEASES):
    return fromgithub.between("owner/repo", installed, latest, lambda repo: releases)


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

    def test_a_nightly_before_a_release_reads_that_releases_notes(self):
        # Audit m25: 0.27.1rc2.dev949 is before 0.27.1, so 0.27.1's notes count.
        releases = [{"tag_name": "v0.27.1", "name": "0.27.1", "body": "final"},
                    {"tag_name": "v0.27.0", "name": "0.27.0", "body": "old"}]
        notes, warning = between("0.27.1rc2.dev949+geac636a7f", "0.27.1", releases)
        self.assertIn("final", notes)
        self.assertNotIn("old", notes)
        self.assertEqual(warning, "")

    def test_build_tags_still_compare_by_their_count(self):
        releases = [{"tag_name": "b10500", "name": "b10500", "body": "new"},
                    {"tag_name": "b10448", "name": "b10448", "body": "here"}]
        notes, _ = between("b10448", "b10500", releases)
        self.assertIn("new", notes)
        self.assertNotIn("here", notes)

    def test_nothing_newer_says_so(self):
        notes, warning = between("v0.4.0", "v0.4.0")
        self.assertEqual(notes, "")
        self.assertIn("No release notes", warning)

    def test_an_unreachable_service_is_a_warning(self):
        def down(repo):
            raise ValueError("down")
        self.assertEqual(fromgithub.between("o/r", "v1", "v2", down),
                         ("", "Could not read the release notes: down"))


if __name__ == "__main__":
    unittest.main()
