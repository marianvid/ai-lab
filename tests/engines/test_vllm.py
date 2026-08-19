import unittest

from ai_lab.engines.vllm import VllmEngine
from ai_lab.types import Format, ModelFile, ModelSet


def model(format=Format.NVFP4, complete=True, entrypoint="/models/nvfp4/a"):
    return ModelSet(id="nvfp4/a", name="a", format=format, entrypoint=entrypoint,
                    files=(ModelFile(entrypoint + "/model.safetensors", 10),),
                    complete=complete,
                    missing=() if complete else ("model-00002-of-00002.safetensors",))


class PlanTests(unittest.TestCase):
    def setUp(self):
        self.engine = VllmEngine(binary="/opt/ai/vllm/.venv/bin/vllm")

    def argv(self, params=None):
        return self.engine.plan(model(), 8082, params or {}).argv

    def flag(self, name, params=None):
        argv = self.argv(params)
        return argv[argv.index(name) + 1]

    def test_it_serves_the_model_directory(self):
        # Unlike GGUF, safetensors models are a directory and vLLM is handed
        # the whole thing.
        argv = self.argv()
        self.assertEqual(argv[:3], ["/opt/ai/vllm/.venv/bin/vllm", "serve",
                                    "/models/nvfp4/a"])

    def test_defaults_are_applied(self):
        self.assertEqual(self.flag("--max-model-len"), "32768")
        self.assertEqual(self.flag("--gpu-memory-utilization"), "0.9")
        self.assertEqual(self.flag("--max-num-seqs"), "32")

    def test_settings_reach_the_command_line(self):
        self.assertEqual(self.flag("--max-model-len", {"context_size": 8192}), "8192")
        self.assertEqual(
            self.flag("--gpu-memory-utilization", {"gpu_memory_fraction": 0.6}), "0.6")
        self.assertEqual(self.flag("--max-num-seqs", {"max_sequences": 8}), "8")

    def test_text_only_is_a_switch_not_a_value(self):
        self.assertNotIn("--language-model-only", self.argv())
        self.assertIn("--language-model-only", self.argv({"language_model_only": True}))

    def test_the_port_is_passed_through(self):
        self.assertEqual(self.flag("--port"), "8082")

    def test_the_model_name_is_what_clients_ask_for(self):
        self.assertEqual(self.flag("--served-model-name"), "a")

    def test_invalid_settings_are_refused(self):
        with self.assertRaises(ValueError):
            self.argv({"context_size": 10})
        with self.assertRaises(ValueError):
            self.argv({"gpu_memory_fraction": 1.5})

    def test_unknown_settings_are_refused(self):
        with self.assertRaises(ValueError):
            self.argv({"cache_type_k": "q4_0"})

    def test_a_gguf_model_is_refused(self):
        with self.assertRaises(ValueError) as caught:
            self.engine.plan(model(format=Format.GGUF), 8082, {})
        self.assertIn("gguf", str(caught.exception))

    def test_an_incomplete_model_is_refused(self):
        with self.assertRaises(ValueError) as caught:
            self.engine.plan(model(complete=False), 8082, {})
        self.assertIn("missing", str(caught.exception))

    def test_it_reads_the_formats_that_matter_on_blackwell(self):
        formats = self.engine.formats()
        self.assertIn(Format.NVFP4, formats)
        self.assertIn(Format.FP8, formats)
        self.assertNotIn(Format.GGUF, formats)

    def test_it_serves_no_page(self):
        self.assertFalse(self.engine.plan(model(), 8082, {}).web_ui)


class ParamTests(unittest.TestCase):
    def test_every_setting_reserves_memory(self):
        # vLLM takes temperature and the rest from each request, so there is
        # nothing to fix at startup and no generation group.
        engine = VllmEngine(binary="/bin/vllm")
        self.assertTrue(all(spec.group == "memory" for spec in engine.params()))


if __name__ == "__main__":
    unittest.main()
