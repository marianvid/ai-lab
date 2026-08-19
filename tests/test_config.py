import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from ai_lab.config import Config, ConfigStore, Instance, Repository


class ConfigStoreTests(unittest.TestCase):
    def setUp(self):
        self._temporary = TemporaryDirectory()
        self.path = Path(self._temporary.name) / "config.json"
        self.addCleanup(self._temporary.cleanup)

    def write(self, payload: dict) -> ConfigStore:
        self.path.write_text(json.dumps(payload))
        return ConfigStore(self.path)

    def test_reads_repositories_and_instances(self):
        store = self.write({
            "title": "AI-Lab",
            "repositories": [{"id": "gguf", "name": "GGUF", "path": "/models/gguf",
                              "format": "gguf"}],
            "instances": [{"id": "qwen", "name": "Coding", "engine": "llamacpp",
                           "model_id": "gguf/qwen", "port": 8080,
                           "params": {"context_size": 4096}}],
        })
        config = store.load()
        self.assertEqual(config.repository("gguf").format, "gguf")
        self.assertEqual(config.instance("qwen").params["context_size"], 4096)

    def test_unknown_lookups_raise(self):
        config = self.write({}).load()
        with self.assertRaises(KeyError):
            config.repository("missing")
        with self.assertRaises(KeyError):
            config.instance("missing")

    def test_round_trip_preserves_everything(self):
        store = ConfigStore(self.path)
        original = Config(
            repositories=[Repository(id="a", name="A", path="/a", format="gguf")],
            instances=[Instance(id="i", name="I", engine="llamacpp",
                                model_id="a/m", port=9000, params={"x": 1})],
        )
        store.save(original)
        self.assertEqual(store.load(), original)

    def test_mutate_writes_back(self):
        store = ConfigStore(self.path)
        store.save(Config())
        with store.mutate() as config:
            config.instances.append(Instance(id="new", name="New", engine="llamacpp",
                                             model_id="a/m", port=8080))
        self.assertEqual(len(store.load().instances), 1)

    def test_mutate_discards_changes_when_the_block_fails(self):
        store = ConfigStore(self.path)
        store.save(Config(title="original"))
        with self.assertRaises(ValueError):
            with store.mutate() as config:
                config.title = "changed"
                raise ValueError("rejected")
        self.assertEqual(store.load().title, "original")

    def test_save_leaves_no_temporary_file_behind(self):
        store = ConfigStore(self.path)
        store.save(Config())
        self.assertEqual([item.name for item in self.path.parent.iterdir()],
                         ["config.json"])


if __name__ == "__main__":
    unittest.main()
