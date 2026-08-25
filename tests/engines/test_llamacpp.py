import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from ai_lab.engines.llamacpp import LlamaCppEngine
from ai_lab.types import Format, ModelFile, ModelSet, Task


def model(format=Format.GGUF, complete=True, entrypoint="/models/a-00001-of-00002.gguf"):
    return ModelSet(id="repo/a", name="a", format=format, entrypoint=entrypoint,
                    files=(ModelFile(entrypoint, 10),), complete=complete,
                    missing=() if complete else ("a-00002-of-00002",))


class PlanTests(unittest.TestCase):
    def setUp(self):
        self.engine = LlamaCppEngine(binary="/bin/llama-server")

    def argv(self, params=None):
        return self.engine.plan(model(), 8080, params or {}).argv

    def flag(self, name, params=None):
        argv = self.argv(params)
        return argv[argv.index(name) + 1]

    def test_defaults_are_applied(self):
        self.assertEqual(self.flag("--ctx-size"), "32768")
        self.assertEqual(self.flag("--cache-type-k"), "q4_0")
        self.assertEqual(self.flag("--flash-attn"), "on")

    def test_settings_reach_the_command_line(self):
        self.assertEqual(self.flag("--ctx-size", {"context_size": 8192}), "8192")
        self.assertEqual(self.flag("--flash-attn", {"flash_attention": False}), "off")

    def test_the_first_shard_is_the_entrypoint(self):
        self.assertEqual(self.flag("--model"), "/models/a-00001-of-00002.gguf")

    def test_invalid_settings_are_refused(self):
        with self.assertRaises(ValueError):
            self.argv({"context_size": 10})

    def test_a_non_gguf_model_is_refused(self):
        with self.assertRaises(ValueError) as caught:
            self.engine.plan(model(format=Format.SAFETENSORS), 8080, {})
        self.assertIn("safetensors", str(caught.exception))

    def test_an_incomplete_model_is_refused(self):
        with self.assertRaises(ValueError) as caught:
            self.engine.plan(model(complete=False), 8080, {})
        self.assertIn("missing", str(caught.exception))

    def test_it_reads_only_gguf(self):
        self.assertEqual(self.engine.formats(), frozenset({Format.GGUF}))

    def test_it_only_generates_text(self):
        self.assertEqual(self.engine.tasks(), frozenset({Task.TEXT_GENERATION}))


if __name__ == "__main__":
    unittest.main()


class SplitAcrossCpuTests(unittest.TestCase):
    """Running a model larger than the card, with part of it in system memory."""

    def setUp(self):
        self.engine = LlamaCppEngine(binary="/bin/llama-server")

    def plan(self, params=None):
        return self.engine.plan(model(), 8080, params or {})

    def flag(self, name, params=None):
        argv = self.plan(params).argv
        return argv[argv.index(name) + 1]

    def test_the_whole_model_goes_on_the_card_by_default(self):
        self.assertEqual(self.flag("--n-gpu-layers"), "999")
        self.assertFalse(self.plan().splits_across_cpu)

    def test_a_layer_count_reaches_the_command_line(self):
        self.assertEqual(self.flag("--n-gpu-layers", {"gpu_layers": 60}), "60")

    def test_a_layer_count_is_declared_as_a_split(self):
        # This is what tells the manager to stop refusing a model bigger than
        # the free memory on the card.
        self.assertTrue(self.plan({"gpu_layers": 60}).splits_across_cpu)

    def test_llama_cpp_can_be_left_to_work_out_the_count(self):
        # The flag must be absent, not set to something. llama.cpp refuses to
        # measure free memory once the flag is present: "n_gpu_layers already
        # set by user, abort".
        plan = self.plan({"gpu_layers": -2})
        self.assertNotIn("--n-gpu-layers", plan.argv)
        self.assertTrue(plan.splits_across_cpu)

    def test_no_layers_on_the_card_is_still_a_split(self):
        # Everything in system memory is legitimate, and very slow.
        self.assertEqual(self.flag("--n-gpu-layers", {"gpu_layers": 0}), "0")
        self.assertTrue(self.plan({"gpu_layers": 0}).splits_across_cpu)

    def test_the_all_layers_sentinel_is_not_a_split(self):
        # 999 is what an older version of this project stored to mean "all of
        # them". Reading it as a split would drop the check that catches a model
        # too large to load.
        self.assertEqual(self.flag("--n-gpu-layers", {"gpu_layers": 999}), "999")
        self.assertFalse(self.plan({"gpu_layers": 999}).splits_across_cpu)


class WebUiTests(unittest.TestCase):
    """llama.cpp ships a chat page as static files, not inside the binary.

    Without --path the server answers the API and nothing else, which is what a
    bare GET / replying 415 was telling us.
    """

    def setUp(self):
        self._temporary = TemporaryDirectory()
        # Resolved because the engine resolves: on macOS /var is a symlink.
        self.root = Path(self._temporary.name).resolve()
        self.addCleanup(self._temporary.cleanup)
        (self.root / "build" / "bin").mkdir(parents=True)
        self.binary = self.root / "build" / "bin" / "llama-server"
        self.binary.write_text("")

    def build_ui(self):
        dist = self.root / "build" / "tools" / "ui" / "dist"
        dist.mkdir(parents=True)
        (dist / "index.html").write_text("<!doctype html>")
        return dist

    def test_the_page_is_served_when_it_was_built(self):
        dist = self.build_ui()
        engine = LlamaCppEngine(binary=str(self.binary))
        self.assertEqual(engine.web_ui(), str(dist))
        plan = engine.plan(model(), 8080, {})
        self.assertIn("--path", plan.argv)
        self.assertEqual(plan.argv[plan.argv.index("--path") + 1], str(dist))
        self.assertTrue(plan.web_ui)

    def test_nothing_is_claimed_when_it_was_not_built(self):
        engine = LlamaCppEngine(binary=str(self.binary))
        self.assertIsNone(engine.web_ui())
        plan = engine.plan(model(), 8080, {})
        self.assertNotIn("--path", plan.argv)
        self.assertFalse(plan.web_ui)


class ReasoningTests(unittest.TestCase):
    """Three separate controls, easily confused for one another.

    Setting the budget to 0 looked like an off switch and was not: it closes
    the thinking block immediately, and the model writes the same thoughts as
    its answer instead. The switch is --reasoning.
    """

    def setUp(self):
        self.engine = LlamaCppEngine(binary="/bin/llama-server")

    def argv(self, params=None):
        return self.engine.plan(model(), 8080, params or {}).argv

    def test_thinking_is_left_to_the_model_by_default(self):
        self.assertNotIn("--reasoning", self.argv())

    def test_thinking_can_be_turned_off(self):
        argv = self.argv({"reasoning": "off"})
        self.assertEqual(argv[argv.index("--reasoning") + 1], "off")

    def test_effort_is_left_to_the_model_by_default(self):
        self.assertNotIn("--reasoning-effort", self.argv())

    def test_effort_can_be_asked_for(self):
        argv = self.argv({"reasoning_effort": "high"})
        self.assertEqual(argv[argv.index("--reasoning-effort") + 1], "high")

    def test_the_budget_is_separate_from_the_switch(self):
        argv = self.argv({"reasoning_budget": 512})
        self.assertEqual(argv[argv.index("--reasoning-budget") + 1], "512")
        self.assertNotIn("--reasoning", argv)

    def test_a_nonsense_level_is_refused(self):
        with self.assertRaises(ValueError):
            self.argv({"reasoning_effort": "enormous"})
