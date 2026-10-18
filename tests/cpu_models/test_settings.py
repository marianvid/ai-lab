"""The `cpu_models` section: what is accepted and what is refused, with the reason."""

import unittest

from ai_lab.config.store import Config
from ai_lab.cpu_models.settings import CpuModelsConfig

BINARY = "/opt/ai/llama.cpp/build/bin/llama-server"
BGE = {"path": "/models/embeddings/bge-m3-FP16.gguf", "pooling": "cls", "port": 8390}


class CpuModelsConfigTests(unittest.TestCase):
    def test_a_model_gets_the_measured_defaults(self):
        config = CpuModelsConfig.from_mapping({"binary": BINARY, "models": {"bge-m3": BGE}})
        model = config.models["bge-m3"]
        self.assertEqual((model.parallel, model.threads, model.context, model.idle_stop_s),
                         (4, 12, 8192, 1800.0))
        self.assertEqual(config.binary, BINARY)

    def test_absent_or_empty_means_no_models(self):
        self.assertEqual(CpuModelsConfig.from_mapping(None).models, {})
        self.assertEqual(CpuModelsConfig.from_mapping({}).models, {})

    def test_mistakes_are_refused_by_name(self):
        cases = [
            ("not an object", "cpu_models must be an object"),
            ({"binary": BINARY, "models": {"m": "x"}}, "models.m must be an object"),
            ({"binary": BINARY, "models": {"m": {**BGE, "path": "rel.gguf"}}}, "absolute path"),
            ({"binary": BINARY, "models": {"m": {**BGE, "pooling": "max"}}}, "pooling"),
            ({"binary": BINARY, "models": {"m": {**BGE, "port": 80}}}, "port"),
            ({"binary": BINARY, "models": {"m": {**BGE, "parallel": True}}}, "parallel"),
            ({"binary": BINARY, "models": {"m": {**BGE, "idle_stop_s": -1}}}, "idle_stop_s"),
            ({"binary": "llama-server", "models": {"m": BGE}}, "binary"),
            ({"binary": BINARY, "models": {"a": BGE, "b": BGE}}, "share a port"),
            ({"binary": BINARY, "models": {"m": {**BGE, "card": 1}}}, "card must be an object"),
            ({"binary": BINARY, "models": {"m": {**BGE, "kv_unified": "yes"}}}, "true or false"),
            ({"binary": BINARY, "models": {"m": {**BGE, "card": {"port": 8390,
                                                                 "needs_mb": 1300}}}},
             "must differ"),
            ({"binary": BINARY, "models": {"m": {**BGE, "card": {"port": 8391}}}}, "needs_mb"),
            ({"binary": BINARY, "models": {"a": {**BGE, "card": {"port": 8391, "needs_mb": 1}},
                                           "b": {**BGE, "port": 8391}}}, "share a port"),
        ]
        for raw, words in cases:
            with self.subTest(raw=raw), self.assertRaisesRegex(ValueError, words):
                CpuModelsConfig.from_mapping(raw)  # type: ignore[arg-type]

    def test_a_card_entry_gives_a_second_port_and_the_room_it_needs(self):
        raw = {**BGE, "card": {"port": 8391, "needs_mb": 1300}}
        model = CpuModelsConfig.from_mapping({"binary": BINARY, "models": {"m": raw}}).models["m"]
        self.assertEqual(model.ports, (8390, 8391))
        self.assertIsNotNone(model.card)
        self.assertEqual(model.card.room_mb if model.card else 0, 1800)

    def test_the_configuration_carries_the_section(self):
        config = Config(cpu_models={"binary": BINARY, "models": {"bge-m3": BGE}})
        self.assertEqual(list(config.cpu_model_policy.models), ["bge-m3"])


if __name__ == "__main__":
    unittest.main()
