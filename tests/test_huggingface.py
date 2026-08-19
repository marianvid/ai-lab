import unittest

from ai_lab.downloads.huggingface import HuggingFaceClient, RemoteFile, group


class GroupTests(unittest.TestCase):
    def files(self, *pairs):
        return [RemoteFile(path=path, size_bytes=size) for path, size in pairs]

    def test_one_quantisation_is_one_set(self):
        sets = group("org/model", self.files(("Q4_K_M.gguf", 100)))
        self.assertEqual(len(sets), 1)
        self.assertEqual(sets[0].name, "Q4_K_M")
        self.assertEqual(sets[0].format, "gguf")

    def test_several_quantisations_become_several_sets(self):
        """One repository commonly holds the same model at different sizes."""
        sets = group("org/model", self.files(("Q4_K_M.gguf", 100), ("Q8_0.gguf", 200)))
        self.assertEqual([item.name for item in sets], ["Q4_K_M", "Q8_0"])

    def test_shards_of_one_model_stay_together(self):
        sets = group("org/model", self.files(
            ("big-00001-of-00002.gguf", 50), ("big-00002-of-00002.gguf", 50)))
        self.assertEqual(len(sets), 1)
        self.assertEqual(sets[0].size_bytes, 100)
        self.assertTrue(sets[0].complete)

    def test_companions_ride_along(self):
        sets = group("org/model", self.files(
            ("model-00001-of-00001.safetensors", 100),
            ("tokenizer.json", 5), ("config.json", 1)))
        self.assertEqual(len(sets), 1)
        self.assertEqual({item.path for item in sets[0].files},
                         {"model-00001-of-00001.safetensors", "tokenizer.json", "config.json"})

    def test_a_repository_missing_a_shard_is_flagged(self):
        sets = group("org/model", self.files(("big-00001-of-00003.gguf", 50)))
        self.assertFalse(sets[0].complete)
        self.assertIn("big-00002-of-00003", sets[0].missing)

    def test_subdirectories_are_separate_models(self):
        sets = group("org/model", self.files(("Q4/model.gguf", 10), ("Q8/model.gguf", 20)))
        self.assertEqual([item.name for item in sets], ["Q4/model", "Q8/model"])

    def test_readme_and_licence_are_not_models(self):
        self.assertEqual(group("org/model", self.files(("README.md", 1), (".gitattributes", 1))), [])

    def test_download_url_is_escaped(self):
        item = RemoteFile(path="folder/my model.gguf", size_bytes=1)
        self.assertEqual(item.url("org/repo"),
                         "https://huggingface.co/org/repo/resolve/main/folder/my%20model.gguf")


class ClientTests(unittest.TestCase):
    def test_search_maps_the_api_shape(self):
        client = HuggingFaceClient(opener=lambda url: [
            {"modelId": "org/model-GGUF", "downloads": 12, "likes": 3,
             "lastModified": "2026-01-01", "tags": ["transformers", "gguf"]}])
        result = client.search("qwen")[0]
        self.assertEqual(result["repo"], "org/model-GGUF")
        self.assertEqual(result["formats"], ["gguf"])

    def test_an_older_gguf_repository_is_recognised_from_its_name(self):
        client = HuggingFaceClient(opener=lambda url: [
            {"modelId": "org/model-GGUF", "tags": []}])
        self.assertEqual(client.search("model")[0]["formats"], ["gguf"])

    def test_search_skips_the_network_for_an_empty_query(self):
        def explode(url):
            raise AssertionError("should not have been called")
        self.assertEqual(HuggingFaceClient(opener=explode).search("   "), [])

    def test_file_sizes_come_from_lfs_when_absent(self):
        """Large weights are stored in LFS, where the size lives on a sub-object."""
        client = HuggingFaceClient(opener=lambda url: [
            {"type": "file", "path": "a.gguf", "lfs": {"size": 999}},
            {"type": "directory", "path": "sub"},
        ])
        files = client.files("org/model")
        self.assertEqual(len(files), 1)
        self.assertEqual(files[0].size_bytes, 999)


if __name__ == "__main__":
    unittest.main()
