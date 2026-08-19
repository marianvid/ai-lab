import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from ai_lab.catalog import Catalog
from ai_lab.naming import split_shard
from ai_lab.config import Repository
from ai_lab.types import Format

from tests.support import make_files, repository


class ShardNameTests(unittest.TestCase):
    def test_recognises_a_shard(self):
        self.assertEqual(split_shard("model-00002-of-00005"), ("model", 2, 5))

    def test_keeps_hyphenated_names_intact(self):
        self.assertEqual(split_shard("Qwen3-35B-A3B-00001-of-00003"),
                         ("Qwen3-35B-A3B", 1, 3))

    def test_a_plain_name_is_not_a_shard(self):
        self.assertIsNone(split_shard("Qwen3-35B-Q4_K_M"))


class CatalogTests(unittest.TestCase):
    def setUp(self):
        self._temporary = TemporaryDirectory()
        self.root = Path(self._temporary.name)
        self.addCleanup(self._temporary.cleanup)

    def scan(self, format="gguf"):
        return Catalog().scan([repository(self.root, format=format)])

    def test_a_single_file_is_one_model(self):
        make_files(self.root / "qwen-coder", "Qwen3-Q4_K_M.gguf", size=100)
        models = self.scan()
        self.assertEqual(len(models), 1)
        self.assertEqual(models[0].name, "Qwen3-Q4_K_M")
        self.assertEqual(models[0].id, "repo/qwen-coder/Qwen3-Q4_K_M")
        self.assertEqual(models[0].size_bytes, 100)
        self.assertTrue(models[0].complete)

    def test_shards_collapse_into_one_model(self):
        make_files(self.root / "big", "big-00001-of-00003.gguf",
                   "big-00002-of-00003.gguf", "big-00003-of-00003.gguf", size=50)
        models = self.scan()
        self.assertEqual(len(models), 1)
        self.assertEqual(len(models[0].files), 3)
        self.assertEqual(models[0].size_bytes, 150)
        self.assertTrue(models[0].complete)

    def test_a_missing_shard_is_reported_not_hidden(self):
        make_files(self.root / "big", "big-00001-of-00003.gguf",
                   "big-00003-of-00003.gguf")
        models = self.scan()
        self.assertEqual(len(models), 1)
        self.assertFalse(models[0].complete)
        self.assertEqual(models[0].missing, ("big-00002-of-00003",))

    def test_unrelated_models_in_one_directory_stay_separate(self):
        make_files(self.root / "mixed", "alpha.gguf", "beta.gguf")
        self.assertEqual([item.name for item in self.scan()], ["alpha", "beta"])

    def test_companions_join_the_model(self):
        make_files(self.root / "gemma", "model-00001-of-00002.safetensors",
                   "model-00002-of-00002.safetensors", "config.json",
                   "tokenizer.json", "tokenizer_config.json")
        models = self.scan(format="safetensors")
        names = sorted(Path(item.path).name for item in models[0].files)
        self.assertIn("tokenizer.json", names)
        self.assertIn("config.json", names)
        self.assertEqual(len(models[0].files), 5)

    def test_unrelated_files_are_ignored(self):
        make_files(self.root / "qwen", "model.gguf", "README.md", "notes.txt")
        self.assertEqual(len(self.scan()[0].files), 1)

    def test_gguf_entrypoint_is_the_first_shard(self):
        make_files(self.root / "big", "big-00001-of-00002.gguf",
                   "big-00002-of-00002.gguf")
        self.assertTrue(self.scan()[0].entrypoint.endswith("big-00001-of-00002.gguf"))

    def test_safetensors_entrypoint_is_the_directory(self):
        make_files(self.root / "gemma", "model.safetensors", "config.json")
        models = self.scan(format="safetensors")
        self.assertEqual(models[0].entrypoint, str(self.root / "gemma"))
        self.assertEqual(models[0].format, Format.SAFETENSORS)

    def test_a_missing_repository_directory_is_skipped(self):
        absent = Repository(id="gone", name="Gone", path="/does/not/exist", format="gguf")
        self.assertEqual(Catalog().scan([absent]), [])

    def test_find_raises_for_an_unknown_model(self):
        make_files(self.root / "qwen", "model.gguf")
        repositories = [repository(self.root)]
        self.assertEqual(Catalog().find(repositories, "repo/qwen/model").name, "model")
        with self.assertRaises(KeyError):
            Catalog().find(repositories, "repo/nope")


if __name__ == "__main__":
    unittest.main()


class DirectoryIsTheModelTests(unittest.TestCase):
    """Safetensors spreads one model over a directory; GGUF puts one in a file."""

    def build(self, format, files):
        self.tmp = TemporaryDirectory()
        root = Path(self.tmp.name)
        (root / "a-model").mkdir()
        for name in files:
            (root / "a-model" / name).write_bytes(b"x")
        self.addCleanup(self.tmp.cleanup)
        return Catalog().scan([Repository(id="repo", name="R",
                                          path=str(root), format=format)])

    def test_a_safetensors_directory_is_one_model_named_after_itself(self):
        models = self.build("nvfp4", [
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
            "model.safetensors.index.json",
            "config.json",
        ])
        self.assertEqual(len(models), 1)
        self.assertEqual(models[0].name, "a-model")
        self.assertEqual(models[0].id, "repo/a-model")

    def test_extra_weight_files_are_parts_not_separate_models(self):
        """model_mtp and model-towers are components; alone they will not load."""
        models = self.build("nvfp4", [
            "model.safetensors",
            "model_mtp.safetensors",
            "model-towers.safetensors",
            "config.json",
        ])
        self.assertEqual(len(models), 1)
        self.assertEqual(models[0].name, "a-model")
        self.assertEqual(len(models[0].files), 4)

    def test_gguf_still_gets_one_model_per_file(self):
        models = self.build("gguf", ["alpha.gguf", "beta.gguf"])
        self.assertEqual(sorted(m.name for m in models), ["alpha", "beta"])
