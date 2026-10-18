"""The wheel check finds a missing application file and a private one."""

import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

from ai_lab.wheel_check import application_files, differences


class WheelCheckTests(unittest.TestCase):
    def setUp(self):
        self.source = Path(tempfile.mkdtemp())
        (self.source / "ai_lab" / "web").mkdir(parents=True)
        (self.source / "ai_lab" / "main.py").write_text("")
        (self.source / "ai_lab" / "web" / "app.js").write_text("")
        (self.source / "ai_lab" / "notes.txt").write_text("")
        self.wheel = self.source / "built.whl"

    def build(self, *names):
        with ZipFile(self.wheel, "w") as archive:
            for name in names:
                archive.writestr(name, "")

    def test_only_shipped_kinds_of_file_count(self):
        self.assertEqual(application_files(self.source), {"ai_lab/main.py", "ai_lab/web/app.js"})

    def test_a_complete_wheel_has_no_differences(self):
        self.build("ai_lab/main.py", "ai_lab/web/app.js")
        self.assertEqual(differences(self.wheel, self.source), ([], []))

    def test_missing_and_private_files_are_named(self):
        self.build("ai_lab/main.py", "opts/config.json")
        self.assertEqual(differences(self.wheel, self.source),
                         (["ai_lab/web/app.js"], ["opts/config.json"]))
