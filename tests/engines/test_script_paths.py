"""Every engine's server script is where its command line says it is.

Engines start AI-Lab's own server scripts by path (`ai_lab/<parts>`), so a
module moved without its engine following would only fail when the model is
loaded. This finds it here instead.
"""

import unittest
from pathlib import Path

from ai_lab.engines.isolated import AI_LAB_PACKAGE
from ai_lab.engines.registry import engine_classes
from ai_lab.engines.text.text_launch import default_launcher


class ScriptPathTests(unittest.TestCase):
    def test_every_engine_s_server_script_exists(self):
        checked = 0
        for engine_id, cls in engine_classes().items():
            script = getattr(cls, "script", ())
            if not script:
                continue
            with self.subTest(engine=engine_id):
                self.assertTrue(AI_LAB_PACKAGE.joinpath(*script).is_file(), script)
            checked += 1
        self.assertGreater(checked, 15)

    def test_every_text_launcher_exists(self):
        for engine_id, cls in engine_classes().items():
            launcher = getattr(cls, "launcher", "")
            if launcher:
                with self.subTest(engine=engine_id):
                    self.assertTrue(Path(default_launcher(launcher)).is_file(), launcher)
