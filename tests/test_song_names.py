"""The manager's song names and the YuE2 adapter's own copy are one rule."""

import unittest

from ai_lab.music.backends import yue2_web_backend as adapter
from ai_lab.shared import song_names


class OneRuleTests(unittest.TestCase):
    def test_the_adapters_copy_is_the_same_rule(self):
        self.assertEqual(song_names.REQUEST_ID.pattern, adapter.REQUEST_ID.pattern)
        self.assertEqual(song_names.GENERATION_ID.pattern, adapter.GENERATION_ID.pattern)

    def test_a_name_that_could_leave_its_folder_is_refused_on_both_sides(self):
        for bad in ("../a", ".hidden", "a/b", "", None, "x" * 181):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    song_names.check_generation_id(bad)
                with self.assertRaises(ValueError):
                    adapter.check_generation_id(bad)
        self.assertEqual(song_names.check_generation_id("song-a.v1"), "song-a.v1")


if __name__ == "__main__":
    unittest.main()
