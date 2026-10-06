"""The per-engine option rules behind `validate_configuration`."""
import unittest

from ai_lab.config import Config, Instance, ModelRoot, Repository
from ai_lab.config_validation import (ENGINE_RULES, absolute_path, file_name, file_names,
                                      number, one_of, options_pass, positive, text, whole,
                                      validate_configuration)

VALID = {
    "kokoro": {"language_code": "a", "default_voice": "af", "repo_id": "x/y"},
    "voxcpm": {"cfg_value": 2.0, "inference_timesteps": 10},
    "khala": {"default_bucket": 2, "maximum_bucket": 8},
    "higgs": {"worker_port": 9000, "memory_reservation_mb": 1, "mem_fraction_static": 0.5},
    "higgs_local": {"memory_reservation_mb": 1},
    "heartmula": {"version": "3B", "checkpoint_subdir": "ck", "topk": 50,
                  "temperature": 1.0, "cfg_scale": 1.5, "memory_reservation_mb": 1},
    "yue2": {"cot": "full", "memory_budget_gib": 24, "memory_reservation_mb": 1},
    "comfy_music": {"workflow": "/w.json", "component_subdirs": ["vae"],
                    "memory_reservation_mb": 1},
    "mulacover": {"checkpoint_subdir": "ck", "topk": 50, "temperature": 1.0,
                  "cfg_scale": 1.5, "memory_reservation_mb": 1},
    "levo2": {"lm": "a.gguf", "flow": "b.gguf", "vae": "c.gguf", "steps": 50,
              "cfg": 1.5, "memory_reservation_mb": 1},
    "stableaudio3": {"steps": 8, "cfg_scale": 1.0, "memory_reservation_mb": 1},
}


class EngineRuleTests(unittest.TestCase):
    def test_every_engine_has_a_valid_example_that_passes(self):
        self.assertEqual(set(VALID), set(ENGINE_RULES))
        for engine, options in VALID.items():
            with self.subTest(engine=engine):
                self.assertTrue(options_pass(engine, options))

    def test_dropping_any_option_fails(self):
        for engine, options in VALID.items():
            for key in options:
                with self.subTest(engine=engine, key=key):
                    broken = {k: v for k, v in options.items() if k != key}
                    self.assertFalse(options_pass(engine, broken))

    def test_options_must_be_an_object(self):
        self.assertFalse(options_pass("yue2", "full"))
        self.assertTrue(options_pass("an-engine-without-rules", "anything"))


class SingleRuleTests(unittest.TestCase):
    def test_text_and_names(self):
        self.assertFalse(text("k")({"k": ""}))
        self.assertTrue(file_name("k")({"k": "model"}))
        self.assertFalse(file_name("k")({"k": "a/model"}))
        self.assertTrue(file_names("k")({"k": ["a", "b"]}))
        self.assertFalse(file_names("k")({"k": []}))
        self.assertTrue(absolute_path("k")({"k": "/x"}))
        self.assertFalse(absolute_path("k")({"k": "x"}))

    def test_numbers_exclude_booleans_and_honour_open_ends(self):
        self.assertFalse(whole("k", 0, 10)({"k": True}))
        self.assertFalse(whole("k", 0, 10)({"k": 1.0}))
        self.assertTrue(number("k", 0, 1)({"k": 0}))
        self.assertFalse(number("k", 0, 1, above_low=True)({"k": 0}))
        self.assertFalse(number("k", 0, 1, below_high=True)({"k": 1}))
        self.assertFalse(positive("k")({"k": 0}))

    def test_one_of_refuses_unhashable_values_instead_of_crashing(self):
        self.assertFalse(one_of("k", {"a"})({"k": ["a"]}))
        self.assertTrue(one_of("k", {"a"})({"k": "a"}))


class HiggsWorkerPortTests(unittest.TestCase):
    def config(self, worker_port):
        return Config(
            port=8090, models_root="/models",
            model_roots=[ModelRoot("core", "Core", "/models")],
            repositories=[Repository("tts", "TTS", "safetensors", task="speech-synthesis")],
            instances=[Instance("voice", "higgs", "tts/h", 8100)],
            engines={"higgs": {"model_options": {"h": {
                "worker_port": worker_port, "memory_reservation_mb": 1,
                "mem_fraction_static": 0.5}}}},
        )

    def test_worker_port_must_differ_from_every_other_port(self):
        for taken in (8090, 8100):
            with self.subTest(port=taken):
                with self.assertRaisesRegex(ValueError, "Higgs worker settings are invalid"):
                    validate_configuration(self.config(taken), {"higgs"})
        validate_configuration(self.config(9000), {"higgs"})


if __name__ == "__main__":
    unittest.main()
