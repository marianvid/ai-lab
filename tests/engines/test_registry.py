import unittest

from ai_lab.engines.registry import Registry
from ai_lab.types import Capabilities


def capabilities(engines=frozenset({"llamacpp"}), accelerator="cuda"):
    return Capabilities(supervisor="systemd", engines=engines,
                        accelerator_kind=accelerator)


class RegistryTests(unittest.TestCase):
    def setUp(self):
        self.registry = Registry()

    def test_every_engine_is_known_everywhere(self):
        self.assertEqual(set(self.registry.known()),
                         {"llamacpp", "vllm", "nemo", "onnx", "pyannote"})

    def test_only_installed_engines_are_available(self):
        self.assertEqual(set(self.registry.available(capabilities())), {"llamacpp"})

    def test_an_unavailable_engine_is_listed_with_a_reason(self):
        rows = {row["id"]: row for row in self.registry.describe(capabilities(accelerator="metal"))}
        self.assertFalse(rows["vllm"]["available"])
        self.assertEqual(rows["vllm"]["reason"], "Requires an NVIDIA GPU")
        self.assertTrue(rows["llamacpp"]["available"])

    def test_describe_carries_the_settings_for_the_form(self):
        rows = {row["id"]: row for row in self.registry.describe(capabilities())}
        keys = {item["key"] for item in rows["llamacpp"]["params"]}
        self.assertIn("context_size", keys)

    def test_describe_carries_the_jobs_for_model_filtering(self):
        rows = {row["id"]: row for row in self.registry.describe(capabilities())}
        self.assertEqual(rows["llamacpp"]["tasks"], ["text-generation"])
        self.assertIn("transcription", rows["vllm"]["tasks"])
        self.assertEqual(rows["nemo"]["tasks"],
                         ["diarization", "transcription"])
        self.assertEqual(rows["onnx"]["tasks"], ["vad"])
        self.assertEqual(rows["pyannote"]["tasks"], ["diarization"])
        transcription = rows["vllm"]["task_params"]["transcription"]
        self.assertNotIn("context_size", {item["key"] for item in transcription})

    def test_unknown_engine_raises(self):
        with self.assertRaises(KeyError):
            self.registry.get("nope")


class ConfiguredBinaryTests(unittest.TestCase):
    """A machine can hold two builds of llama.cpp; PATH order is not a decision."""

    def test_a_configured_binary_beats_whatever_is_on_path(self):
        registry = Registry({"llamacpp": {"binary": "/opt/ai/llama.cpp/build/bin/llama-server"}})
        self.assertEqual(registry.get("llamacpp").binary,
                         "/opt/ai/llama.cpp/build/bin/llama-server")

    def test_the_chosen_binary_is_reported_so_it_is_visible(self):
        registry = Registry({"llamacpp": {"binary": "/custom/llama-server"}})
        rows = {row["id"]: row for row in registry.describe(capabilities())}
        self.assertEqual(rows["llamacpp"]["binary"], "/custom/llama-server")

    def test_without_configuration_it_falls_back_to_path(self):
        self.assertTrue(Registry().get("llamacpp").binary.endswith("llama-server"))

    def test_nemo_keeps_its_runtime_and_server_paths_separate(self):
        registry = Registry({"nemo": {
            "binary": "/opt/ai/nemo/.venv/bin/python",
            "server": "/opt/ai-lab/ai_lab/audio/server.py",
        }})
        engine = registry.get("nemo")
        self.assertEqual(engine.binary, "/opt/ai/nemo/.venv/bin/python")
        self.assertEqual(engine.server, "/opt/ai-lab/ai_lab/audio/server.py")

    def test_the_configured_binary_reaches_the_command_line(self):
        from ai_lab.types import Format, ModelFile, ModelSet
        registry = Registry({"llamacpp": {"binary": "/custom/llama-server"}})
        model = ModelSet(id="r/a", name="a", format=Format.GGUF,
                         entrypoint="/models/a.gguf",
                         files=(ModelFile("/models/a.gguf", 1),))
        plan = registry.get("llamacpp").plan(model, 8080, {})
        self.assertEqual(plan.argv[0], "/custom/llama-server")


if __name__ == "__main__":
    unittest.main()
