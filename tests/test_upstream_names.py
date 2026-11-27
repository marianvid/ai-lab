"""The upstream names the HeartMuLa page and the Higgs port are wired through, checked at start."""

import argparse
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from ai_lab.music.servers.heartmula_server import (HEARTMUSE_NAMES, check_heartmuse,
                                                   launch_heartmuse, missing_heartmuse_names)
from ai_lab.speech.higgs_local_backend import MODEL_NAMES, PORT_NAMES, check_port, missing_names


def having(names, *, but=()):
    return SimpleNamespace(**{name: object() for name in names if name not in but})


class HiggsPortTests(unittest.TestCase):
    def test_a_port_with_every_name_is_taken(self):
        self.assertEqual(missing_names(having(MODEL_NAMES), having(PORT_NAMES)), [])

    def test_a_renamed_name_stops_the_load_and_says_what_it_was_written_against(self):
        with self.assertRaisesRegex(RuntimeError, r"lacks model\._decode_codes, "
                                                  r"port\.apply_delay_pattern.*5eab2e3f"):
            check_port(having(MODEL_NAMES, but=("_decode_codes",)),
                       having(PORT_NAMES, but=("apply_delay_pattern",)))


class HeartMuseNamesTests(unittest.TestCase):
    def test_a_missing_name_stops_the_page_and_names_the_checkout_used(self):
        with self.assertRaisesRegex(RuntimeError, r"lacks generator\._pipeline.*a28600c5d6ba"):
            check_heartmuse(generator=having(HEARTMUSE_NAMES["generator"], but=("_pipeline",)))
        self.assertEqual(missing_heartmuse_names(config=having(HEARTMUSE_NAMES["config"]),
                                                 app=having(HEARTMUSE_NAMES["app"])), [])

    def test_the_page_is_wired_around_the_resident_pipeline_in_order(self):
        """A stand-in HeartMuse: its generator reads the variants when it is imported."""
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        folder = Path(temporary.name)
        (folder / "config.py").write_text(
            "MODEL_VARIANTS = {'base': 1, 'large': 2}\nMODEL_VARIANT_LABELS = {}\n"
            "DEFAULT_MODEL_VARIANT = 'large'\nOUTPUT_DIR = '/tmp'\n")
        (folder / "generator.py").write_text(
            "import config\nSEEN = sorted(config.MODEL_VARIANTS)\n_pipeline = None\n"
            "_active_lazy_load = False\n_active_variant = ''\n"
            "def generate_music(*a, **k):\n    return 'song'\n"
            "def unload_pipeline():\n    raise AssertionError('never unloaded')\n")
        (folder / "app.py").write_text(
            "PLAYLIST_JS = ''\nLAUNCHED = []\nclass _App:\n"
            "    def queue(self, **k):\n        return self\n"
            "    def launch(self, **k):\n        LAUNCHED.append(k)\napp = _App()\n")
        self.addCleanup(lambda: [sys.modules.pop(name, None)
                                 for name in ("config", "generator", "app")])
        self.addCleanup(lambda: sys.path.remove(str(folder)) if str(folder) in sys.path else None)
        backend = SimpleNamespace(pipeline="resident", lock=__import__("threading").Lock())
        args = argparse.Namespace(heartmuse_root=folder, bundle_root=folder,
                                  output_root=folder, ui_port=18000)
        launch_heartmuse(backend, args)  # type: ignore[arg-type]
        generator, app = sys.modules["generator"], sys.modules["app"]
        self.assertEqual(generator.SEEN, ["base"])              # narrowed before it was read
        self.assertEqual(generator._pipeline, "resident")
        self.assertEqual(generator.generate_music(), "song")
        generator.unload_pipeline()                             # the resident one stays
        self.assertEqual(app.LAUNCHED[0]["server_port"], 18000)


if __name__ == "__main__":
    unittest.main()
