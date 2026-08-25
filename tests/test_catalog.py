import tempfile
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from ai_lab.catalog import Catalog
from ai_lab.naming import split_shard
from ai_lab.config import Repository
from ai_lab.types import Format, Task

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

    def test_a_nemo_checkpoint_is_a_directory_model(self):
        make_files(self.root / "parakeet", "parakeet.nemo", "config.json")
        models = self.scan(format="nemo")
        self.assertEqual(len(models), 1)
        self.assertEqual(models[0].entrypoint, str(self.root / "parakeet"))
        self.assertEqual(models[0].format, Format.NEMO)

    def test_an_onnx_directory_is_one_model_even_with_several_exports(self):
        make_files(self.root / "silero", "silero.onnx", "silero-fp16.onnx")
        models = self.scan(format="onnx")
        self.assertEqual(len(models), 1)
        self.assertEqual(len(models[0].files), 2)
        self.assertEqual(models[0].format, Format.ONNX)

    def test_a_pyannote_pipeline_is_one_tree_model(self):
        make_files(self.root / "community-1", "config.yaml")
        make_files(self.root / "community-1" / "segmentation", "model.safetensors")
        make_files(self.root / "community-1" / "embedding", "model.safetensors")
        models = self.scan(format="pyannote")
        self.assertEqual(len(models), 1)
        self.assertEqual(models[0].format, Format.PYANNOTE)
        self.assertEqual(models[0].entrypoint, str(self.root / "community-1"))
        self.assertEqual(len(models[0].files), 3)

    def test_the_repository_job_reaches_every_model_it_contains(self):
        make_files(self.root / "whisper", "model.safetensors", "config.json")
        repository = Repository(id="audio", name="Audio", path=str(self.root),
                                format="safetensors", task="transcription")
        self.assertEqual(Catalog().scan([repository])[0].task, Task.TRANSCRIPTION)

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


class VisionProjectorTests(unittest.TestCase):
    """A projector is weights, but it is not a model.

    `mmproj-<model>-f16.gguf` sits beside a GGUF model and holds the part that
    turns a picture into something the model can read. llama.cpp is handed it
    with --mmproj alongside the model; started on its own it serves nothing.

    Left as a model it becomes a library entry that can never be started, and
    an entry pointing at it blocks deleting the real model — the pair that made
    `model_mtp.safetensors` a problem for safetensors. There the
    directory-is-the-model rule solved it. GGUF has no such rule, because there
    one file genuinely is one model, so this has to be said out loud.
    """

    def setUp(self):
        self._temporary = TemporaryDirectory()
        self.root = Path(self._temporary.name)
        self.addCleanup(self._temporary.cleanup)

    def scan(self):
        return Catalog().scan([repository(self.root, format="gguf")])

    def test_a_projector_is_not_a_second_model(self):
        make_files(self.root / "gemma-4-26b-a4b",
                   "gemma-4-26B-A4B-it-Q4_K_M.gguf", size=100)
        make_files(self.root / "gemma-4-26b-a4b",
                   "mmproj-gemma-4-26B-A4B-it-f16.gguf", size=10)
        models = self.scan()
        self.assertEqual([model.name for model in models],
                         ["gemma-4-26B-A4B-it-Q4_K_M"])

    def test_the_projector_still_belongs_to_the_model(self):
        # Attached, not ignored: it has to travel with the model when the whole
        # set is downloaded or deleted.
        make_files(self.root / "qwen", "Qwen-Q4_K_M.gguf", size=100)
        make_files(self.root / "qwen", "mmproj-Qwen-bf16.gguf", size=10)
        model = self.scan()[0]
        self.assertIn("mmproj-Qwen-bf16.gguf",
                      [Path(item.path).name for item in model.files])

    def test_the_model_is_still_what_gets_started(self):
        # The entrypoint must be the model, never the projector, whichever way
        # the two happen to sort.
        make_files(self.root / "qwen", "Qwen-Q4_K_M.gguf", size=100)
        make_files(self.root / "qwen", "mmproj-Qwen-bf16.gguf", size=10)
        self.assertTrue(self.scan()[0].entrypoint.endswith("Qwen-Q4_K_M.gguf"))

    def test_a_projector_on_its_own_is_no_model_at_all(self):
        make_files(self.root / "stray", "mmproj-something-f16.gguf", size=10)
        self.assertEqual(self.scan(), [])

    def test_a_model_merely_starting_with_the_letters_is_untouched(self):
        # The separator is part of the rule. Hiding a real model because its
        # name begins with the same letters is a worse fault than showing a
        # projector, so the match has to be the projector's actual shape.
        make_files(self.root / "odd", "mmprojector-lab-Q4_K_M.gguf", size=100)
        self.assertEqual([model.name for model in self.scan()],
                         ["mmprojector-lab-Q4_K_M"])

    def test_a_projector_named_without_a_model_is_still_a_projector(self):
        make_files(self.root / "plain", "Qwen-Q4_K_M.gguf", size=100)
        make_files(self.root / "plain", "mmproj.gguf", size=10)
        self.assertEqual([model.name for model in self.scan()], ["Qwen-Q4_K_M"])


class CarryingCapabilities(unittest.TestCase):
    """A model found on disk says what it can do, or says nothing at all."""

    class Answers:
        """Stands in for the reader, so this tests the wiring, not the parsing."""

        def __init__(self, answer=frozenset({"tools"})):
            self.answer = answer
            self.asked = []

        def of(self, entrypoint, is_gguf, companions):
            self.asked.append((entrypoint, is_gguf))
            return self.answer

    def _repository(self, root):
        (root / "a-model").mkdir()
        (root / "a-model" / "model.gguf").write_bytes(b"x")
        return Repository(id="r", name="r", path=str(root), format="gguf")

    def test_the_answer_reaches_the_model(self):
        root = Path(tempfile.mkdtemp())
        answers = self.Answers()
        found = Catalog(answers).scan([self._repository(root)])
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].capabilities, frozenset({"tools"}))
        # Asked about what the engine is handed — the weights file for GGUF,
        # not the directory holding it.
        self.assertEqual(len(answers.asked), 1)
        self.assertTrue(answers.asked[0][0].endswith("model.gguf"))
        self.assertTrue(answers.asked[0][1])

    def test_a_catalog_with_nobody_to_ask_still_scans(self):
        root = Path(tempfile.mkdtemp())
        found = Catalog().scan([self._repository(root)])
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].capabilities, frozenset())
