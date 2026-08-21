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


class ToolCallingTests(unittest.TestCase):
    """Letting a model call tools, which needs two flags that go together.

    A model that wants to use a tool does not return structured data. It writes
    text, and every model family writes it differently — Qwen one way, Gemma
    another. vLLM has to be told which of those to expect before it will turn
    that text back into a call, and it refuses "auto" tool choice without both
    the switch and the format.

    Two flags that must both be present or both absent is a pair that gets set
    half-way, so there is one setting here rather than two.
    """

    def setUp(self):
        self.engine = VllmEngine(binary="/opt/ai/vllm/.venv/bin/vllm")

    def argv(self, params=None):
        return self.engine.plan(model(), 8082, params or {}).argv

    def test_off_by_default(self):
        # Most work needs no tools, and turning it on costs a startup flag
        # that has to name a format nobody has chosen yet.
        argv = self.argv()
        self.assertNotIn("--enable-auto-tool-choice", argv)
        self.assertNotIn("--tool-call-parser", argv)

    def test_naming_a_format_turns_it_on(self):
        argv = self.argv({"tool_parser": "qwen3_coder"})
        self.assertIn("--enable-auto-tool-choice", argv)
        self.assertEqual(argv[argv.index("--tool-call-parser") + 1], "qwen3_coder")

    def test_the_switch_and_the_format_are_never_separated(self):
        # vLLM starts, and then refuses every request from an agent, if it has
        # one without the other.
        argv = self.argv({"tool_parser": "gemma4"})
        self.assertEqual(argv.count("--enable-auto-tool-choice"), 1)
        self.assertEqual(argv.count("--tool-call-parser"), 1)

    def test_empty_means_off_rather_than_an_empty_flag(self):
        argv = self.argv({"tool_parser": ""})
        self.assertNotIn("--enable-auto-tool-choice", argv)

    def test_a_format_vllm_does_not_know_is_refused_when_typed(self):
        # Rather than as an engine that will not start, minutes later, with a
        # message about argument parsing.
        with self.assertRaises(ValueError) as caught:
            self.argv({"tool_parser": "qwen-coder"})
        self.assertIn("Tool calling", str(caught.exception))

    def test_the_formats_this_machine_actually_uses_are_offered(self):
        offered = dict((spec.key, spec) for spec in self.engine.params())
        choices = offered["tool_parser"].choices
        for name in ("qwen3_coder", "gemma4", "glm47"):
            self.assertIn(name, choices)


class MemorySettingsTests(unittest.TestCase):
    """The three settings that decide whether a context fits at all."""

    def setUp(self):
        self.engine = VllmEngine(binary="/opt/ai/vllm/.venv/bin/vllm")

    def argv(self, params=None):
        return self.engine.plan(model(), 8082, params or {}).argv

    def test_nothing_extra_on_the_command_line_by_default(self):
        argv = self.argv()
        for flag in ("--kv-cache-dtype", "--enable-prefix-caching",
                     "--enforce-eager"):
            self.assertNotIn(flag, argv)

    def test_the_default_cache_precision_is_left_unsaid(self):
        # "auto" is vLLM's own default. Passing it changes nothing and puts a
        # flag on the command line that was never a choice.
        self.assertNotIn("--kv-cache-dtype", self.argv({"kv_cache_dtype": "auto"}))

    def test_a_smaller_cache_precision_reaches_the_command_line(self):
        argv = self.argv({"kv_cache_dtype": "fp8"})
        self.assertEqual(argv[argv.index("--kv-cache-dtype") + 1], "fp8")

    def test_a_precision_vllm_does_not_know_is_refused_when_typed(self):
        with self.assertRaises(ValueError) as caught:
            self.argv({"kv_cache_dtype": "fp4"})
        self.assertIn("Cache precision", str(caught.exception))

    def test_prefix_caching_is_a_switch(self):
        self.assertIn("--enable-prefix-caching", self.argv({"prefix_caching": True}))
        self.assertNotIn("--enable-prefix-caching", self.argv({"prefix_caching": False}))

    def test_eager_mode_is_a_switch(self):
        self.assertIn("--enforce-eager", self.argv({"enforce_eager": True}))
        self.assertNotIn("--enforce-eager", self.argv({"enforce_eager": False}))

    def test_all_three_together(self):
        argv = self.argv({"kv_cache_dtype": "fp8", "prefix_caching": True,
                          "enforce_eager": True})
        for flag in ("--kv-cache-dtype", "--enable-prefix-caching",
                     "--enforce-eager"):
            self.assertIn(flag, argv)

    def test_every_setting_explains_itself(self):
        # The explanation is the tooltip in the interface. A setting with none
        # is one the reader has to guess at, and these trade against each other
        # on a single card.
        for spec in self.engine.params():
            self.assertTrue(spec.help.strip(), f"{spec.key} has no explanation")
