"""What YuE2 Studio decides about a song, in `songs/studio.py` (audit L61)."""

import tempfile
import unittest
from pathlib import Path

from ai_lab.gateway.errors import ModelNotLoadedError, WindowCallActiveError
from ai_lab.songs.studio import SongFolders


class Engine:
    """A YuE2 engine keeping its songs in `folder`."""

    def __init__(self, folder: Path) -> None:
        self.folder = folder

    def generations_dir(self) -> Path:
        return self.folder

    def editor_port(self, port: int) -> int:
        return port + 10000


class Operations:
    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def configured(self) -> list:
        return []

    def engine(self, _name: str) -> Engine:
        return self._engine


class Gateway:
    """YuE2 not loaded: nothing can be writing a song."""

    def resolve(self, model: str, _configured) -> dict:
        return {"instance_id": model, "engine": "yue2"}

    def where_loaded(self, model: str, _path: str):
        raise ModelNotLoadedError(model, [])


class SongFoldersTests(unittest.TestCase):
    def setUp(self):
        self.here = Path(tempfile.mkdtemp())
        (self.here / "song-1").mkdir()
        self.edited: dict[str, str] = {}
        self.songs = SongFolders(Gateway(), Operations(Engine(self.here)), self.edited.get)
        self.query = {"model": "yue2-3b"}

    def test_with_the_engine_down_a_song_is_removed_from_its_folder(self):
        self.assertEqual(self.songs.delete(self.query, "song-1"), {"deleted": "song-1"})
        self.assertFalse((self.here / "song-1").exists())

    def test_the_source_of_a_window_call_is_busy_and_kept(self):
        self.edited["song-1"] = "call-7"
        with self.assertRaisesRegex(WindowCallActiveError, "call-7"):
            self.songs.refuse_busy(self.query, "song-1")
        with self.assertRaisesRegex(WindowCallActiveError, "call-7"):
            self.songs.delete(self.query, "song-1")
        self.assertTrue((self.here / "song-1").is_dir())

    def test_a_song_not_edited_with_the_studio_down_is_not_busy(self):
        self.assertIsNone(self.songs.studio_port(self.query))
        self.songs.refuse_busy(self.query, "song-1")


if __name__ == "__main__":
    unittest.main()
