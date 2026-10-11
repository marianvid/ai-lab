import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from ai_lab.providers import ProviderRateLimitError, ProviderUnavailableError
from ai_lab.providers.dialects import Outcome
from ai_lab.providers.pool import ProviderPool
from ai_lab.providers.settings import ProvidersConfig
from ai_lab.providers.usage import UsageLog

ANTHROPIC = {"dialect": "claude", "binary": "/opt/cli/claude",
             "launch_spacing_s": 0, "rate_limit_waits_s": [0],
             "rate_limit_retries": 2, "network_waits_s": [5, 15]}
MODELS = {"claude/sonnet": {"vendor": "anthropic", "model": "sonnet"}}
RAW = {"vendors": {"anthropic": ANTHROPIC}, "models": MODELS}


def with_default(effort):
    """The same configuration, with a default reasoning effort."""
    return {"vendors": {"anthropic": {**ANTHROPIC, "default_effort": effort}},
            "models": MODELS}


CHAT = "/v1/chat/completions"
REQUEST = {"model": "claude/sonnet", "messages": [{"role": "user", "content": "Hi"}],
           "reasoning_effort": "low"}


def ok(text="Hello"):
    return Outcome(returncode=0, answer=text, errors="")


def failed(errors):
    return Outcome(returncode=1, answer="", errors=errors)


class ScriptedRunner:
    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def __call__(self, dialect, vendor, model, prompt, effort):
        self.calls.append((type(dialect).__name__, vendor.id, model, prompt, effort))
        return self.outcomes.pop(0)


class ProviderPoolTests(unittest.TestCase):
    def setUp(self):
        self._temporary = TemporaryDirectory()
        self.addCleanup(self._temporary.cleanup)
        self.slept = []

    def pool(self, *outcomes):
        self.runner = ScriptedRunner(*outcomes)
        return ProviderPool(ProvidersConfig.from_mapping(RAW),
                            UsageLog(Path(self._temporary.name)),
                            runner=self.runner, sleep=self.slept.append)

    def test_a_name_is_answered_in_the_openai_shape(self):
        pool = self.pool(ok("Hello"))
        answer = pool.complete(CHAT, REQUEST)
        self.assertEqual(answer["choices"][0]["message"]["content"], "Hello")
        self.assertEqual(self.runner.calls[0][:3], ("ClaudeCli", "anthropic", "sonnet"))
        self.assertEqual(self.runner.calls[0][4], "low")
        self.assertTrue(pool.serves("claude/sonnet"))
        self.assertFalse(pool.serves("qwen36-nvfp4"))

    def test_a_rate_limit_is_retried_on_the_same_model(self):
        pool = self.pool(failed("usage limit reached"), ok())
        pool.complete(CHAT, REQUEST)
        self.assertEqual(len(self.runner.calls), 2)
        today = pool.stats()["anthropic"]["today"]
        self.assertEqual((today["requests"], today["ok"]), (2, 1))
        self.assertEqual(today["failures"], {"usage limit": 1})

    def test_a_rate_limit_that_lasts_is_refused_with_429_meaning(self):
        pool = self.pool(*[failed("429 Too Many Requests")] * 3)
        with self.assertRaisesRegex(ProviderRateLimitError, "allowance is spent"):
            pool.complete(CHAT, REQUEST)
        self.assertEqual(len(self.runner.calls), 3)

    def test_a_fatal_error_is_answered_at_once(self):
        pool = self.pool(failed("Invalid API key · Please run /login"))
        with self.assertRaisesRegex(ProviderUnavailableError, "Invalid API key"):
            pool.complete(CHAT, REQUEST)
        self.assertEqual(len(self.runner.calls), 1)

    def test_network_failures_are_retried_after_each_wait(self):
        network = failed("error sending request for url")
        pool = self.pool(network, network, ok())
        pool.complete(CHAT, REQUEST)
        self.assertEqual(self.slept, [5, 15])

    def test_network_failures_that_outlast_the_waits_fail(self):
        pool = self.pool(*[Outcome(returncode=-1, answer="", errors="timeout",
                                   timed_out=True)] * 3)
        with self.assertRaisesRegex(ProviderUnavailableError, "timeout"):
            pool.complete(CHAT, REQUEST)

    def test_only_the_chat_path_without_streaming(self):
        pool = self.pool()
        with self.assertRaises(ValueError):
            pool.complete("/v1/messages", REQUEST)
        with self.assertRaises(NotImplementedError):
            pool.complete(CHAT, {**REQUEST, "stream": True})

    def test_an_effort_claude_does_not_know_is_refused_before_any_call(self):
        pool = self.pool()
        with self.assertRaisesRegex(ValueError, "low, medium, high, xhigh, max"):
            pool.complete(CHAT, {**REQUEST, "reasoning_effort": "bogus"})
        self.assertEqual(self.runner.calls, [])

    def test_every_effort_claude_knows_reaches_the_cli(self):
        pool = self.pool(*[ok()] * 5)
        for effort in ("low", "medium", "high", "xhigh", "max"):
            pool.complete(CHAT, {**REQUEST, "reasoning_effort": effort})
        self.assertEqual([call[4] for call in self.runner.calls],
                         ["low", "medium", "high", "xhigh", "max"])

    def test_a_request_with_no_effort_gets_the_vendor_default(self):
        self.runner = ScriptedRunner(ok(), ok())
        pool = ProviderPool(ProvidersConfig.from_mapping(with_default("medium")),
                            UsageLog(Path(self._temporary.name)), runner=self.runner)
        without = {key: value for key, value in REQUEST.items() if key != "reasoning_effort"}
        pool.complete(CHAT, without)
        pool.complete(CHAT, {**REQUEST, "reasoning_effort": "high"})
        self.assertEqual([call[4] for call in self.runner.calls], ["medium", "high"])
        self.assertEqual(pool.stats()["anthropic"]["default_effort"], "medium")

    def test_no_default_and_no_effort_leaves_the_choice_to_the_cli(self):
        pool = self.pool(ok())
        pool.complete(CHAT, {key: value for key, value in REQUEST.items()
                             if key != "reasoning_effort"})
        self.assertEqual(self.runner.calls[0][4], "")

    def test_a_default_claude_does_not_know_stops_the_start(self):
        with self.assertRaisesRegex(ValueError, "default_effort"):
            ProviderPool(ProvidersConfig.from_mapping(with_default("medum")),
                         UsageLog(Path(self._temporary.name)))

    def test_a_default_effort_must_be_text(self):
        with self.assertRaisesRegex(ValueError, "must be text"):
            ProvidersConfig.from_mapping(with_default(3))

    def test_a_fenced_json_answer_comes_back_plain_only_when_json_was_asked(self):
        fenced = '```json\n{"answers": []}\n```'
        pool = self.pool(ok(fenced), ok(fenced))
        asked = pool.complete(CHAT, {**REQUEST, "response_format": {"type": "json_object"}})
        plain = pool.complete(CHAT, REQUEST)
        self.assertEqual(asked["choices"][0]["message"]["content"], '{"answers": []}')
        self.assertEqual(plain["choices"][0]["message"]["content"], fenced)

    def test_an_opencode_variant_is_left_to_opencode(self):
        raw = {"vendors": {"cloud": {"dialect": "opencode", "binary": "/opt/cli/opencode",
                                     "launch_spacing_s": 0}},
               "models": {"oc/kimi": {"vendor": "cloud", "model": "moonshot/kimi"}}}
        self.runner = ScriptedRunner(ok())
        pool = ProviderPool(ProvidersConfig.from_mapping(raw),
                            UsageLog(Path(self._temporary.name)), runner=self.runner)
        pool.complete(CHAT, {**REQUEST, "model": "oc/kimi", "reasoning_effort": "turbo"})
        self.assertEqual(self.runner.calls[0][4], "turbo")

    def test_catalogue_and_stats_name_every_model(self):
        pool = self.pool()
        self.assertEqual([row["id"] for row in pool.catalogue()], ["claude/sonnet"])
        self.assertEqual(pool.stats()["anthropic"]["models"], ["claude/sonnet"])

    def test_built_from_configuration(self):
        pool = ProviderPool.from_config(RAW, Path(self._temporary.name))
        self.assertTrue(pool.serves("claude/sonnet"))


if __name__ == "__main__":
    unittest.main()
