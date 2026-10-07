import unittest
from typing import Any

from ai_lab.config import Config
from ai_lab.config_validation import validate_configuration
from ai_lab.providers.settings import ProvidersConfig, VendorLimits

VENDORS = {"anthropic": {"dialect": "claude", "binary": "/opt/cli/claude",
                         "home": "/var/lib/ai-lab/cli/anthropic"},
           "openai": {"dialect": "codex", "binary": "/opt/cli/codex",
                      "concurrency": 12, "rate_limit_waits_s": [1, 2]}}
MODELS = {"claude/sonnet": {"vendor": "anthropic", "model": "sonnet"},
          "codex/terra": {"vendor": "openai", "model": "gpt-5.6-terra"}}


class ProvidersConfigTests(unittest.TestCase):
    def test_an_absent_section_means_no_providers(self):
        config = ProvidersConfig.from_mapping(None)
        self.assertEqual((config.vendors, config.models), ({}, {}))

    def test_defaults_are_the_measured_values(self):
        config = ProvidersConfig.from_mapping({"vendors": VENDORS, "models": MODELS})
        limits = config.vendors["anthropic"].limits
        self.assertEqual(limits, VendorLimits())
        self.assertEqual((limits.concurrency, limits.launch_spacing_s), (6, 0.5))
        self.assertEqual(limits.rate_limit_waits_s, (30.0, 120.0, 300.0))
        self.assertEqual(limits.network_waits_s, (5.0, 15.0))

    def test_configured_limits_win(self):
        config = ProvidersConfig.from_mapping({"vendors": VENDORS, "models": MODELS})
        limits = config.vendors["openai"].limits
        self.assertEqual((limits.concurrency, limits.rate_limit_waits_s), (12, (1.0, 2.0)))
        self.assertEqual(config.models["codex/terra"].model, "gpt-5.6-terra")
        self.assertEqual(config.vendors["openai"].home, "")

    def test_impossible_values_are_refused_by_name(self):
        cases: list[tuple[Any, str]] = [
            ({"vendors": {"x": {"dialect": "gemini", "binary": "/b"}}}, "dialect"),
            ({"vendors": {"x": {"dialect": "claude", "binary": "claude"}}}, "absolute"),
            ({"vendors": {"x": {"dialect": "claude", "binary": "/b", "concurrency": 0}}},
             "concurrency"),
            ({"vendors": {"x": {"dialect": "claude", "binary": "/b", "timeout_s": 0}}},
             "timeout_s"),
            ({"vendors": {"x": {"dialect": "claude", "binary": "/b",
                                "network_waits_s": []}}}, "network_waits_s"),
            ({"vendors": {"x": {"dialect": "claude", "binary": "/b",
                                "launch_spacing_s": True}}}, "launch_spacing_s"),
            ({"vendors": VENDORS, "models": {"m": {"vendor": "nobody", "model": "a"}}},
             "configured vendor"),
            ({"vendors": VENDORS, "models": {"m": {"vendor": "openai", "model": " "}}},
             "model name"),
            ({"vendors": {"x": "claude"}}, "object"),
            ("not a mapping", "providers must be an object"),
        ]
        for raw, words in cases:
            with self.subTest(words=words), self.assertRaisesRegex(ValueError, words):
                ProvidersConfig.from_mapping(raw)

    def test_a_bad_section_stops_the_manager_at_start(self):
        config = Config(providers={"vendors": {"x": {"dialect": "y", "binary": "/b"}}})
        with self.assertRaisesRegex(ValueError, "providers.vendors.x.dialect"):
            validate_configuration(config, set())
        self.assertEqual(Config(providers={"vendors": VENDORS, "models": MODELS})
                         .provider_policy.models["claude/sonnet"].vendor, "anthropic")


if __name__ == "__main__":
    unittest.main()
