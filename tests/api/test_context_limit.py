"""A too-long refusal comes back with its numbers; anything else is untouched."""

import json
import unittest

from ai_lab.api.context_limit import REASON, explained, numbers
from ai_lab.api.passthrough import Passthrough

# vLLM 0.26's own sentences (renderers/params.py, v1/engine/input_processor.py).
BY_TOKENS = ("This model's maximum context length is 98304 tokens. However, you "
             "requested 1000 output tokens and your prompt contains 120000 input "
             "tokens, for a total of 121000 tokens. Please reduce the length of "
             "the input prompt or the number of requested output tokens.")
AT_LEAST = BY_TOKENS.replace("contains 120000", "contains at least 98305")
BY_CHARACTERS = ("This model's maximum context length is 32768 tokens. However, you "
                 "requested 500 output tokens and your prompt contains 900000 "
                 "characters (more than 800000 characters, which is the upper bound "
                 "for 32268 input tokens).")
BY_PROMPT = ("The decoder prompt (length 40000) is longer than the maximum model "
             "length of 32768. Make sure that `max_model_len` is no smaller than "
             "the number of text tokens.")


def refusal(message, status=400, content_type="application/json"):
    body = json.dumps({"error": {"message": message, "type": "BadRequestError", "code": 400}})
    return Passthrough(status, content_type, iter([body.encode()]), {})


def body_of(answer):
    return json.loads(b"".join(answer.chunks))


class NumbersTests(unittest.TestCase):
    def test_prompt_counted_in_tokens(self):
        self.assertEqual(numbers(BY_TOKENS), {"reason": REASON, "context_size": 98304,
                                              "prompt_tokens": 120000,
                                              "max_output_tokens": 1000})

    def test_a_truncated_count_is_still_read(self):
        found = numbers(AT_LEAST)
        assert found is not None
        self.assertEqual(found["prompt_tokens"], 98305)

    def test_prompt_refused_on_characters(self):
        self.assertEqual(numbers(BY_CHARACTERS), {"reason": REASON, "context_size": 32768,
                                                  "prompt_characters": 900000,
                                                  "max_output_tokens": 500})

    def test_prompt_longer_than_the_model(self):
        self.assertEqual(numbers(BY_PROMPT), {"reason": REASON, "context_size": 32768,
                                              "prompt_tokens": 40000})

    def test_any_other_sentence_says_nothing(self):
        self.assertIsNone(numbers("Only user and assistant roles are supported"))


class ExplainedTests(unittest.TestCase):
    def test_the_numbers_are_added_and_the_message_kept(self):
        answer = explained(refusal(BY_TOKENS))
        body = body_of(answer)
        self.assertEqual(answer.status, 400)
        self.assertEqual(body["error"]["message"], BY_TOKENS)
        self.assertEqual(body["ai_lab"]["context_size"], 98304)
        self.assertEqual(body["ai_lab"]["prompt_tokens"], 120000)

    def test_another_refusal_goes_through_as_it_came(self):
        answer = explained(refusal("some other problem"))
        self.assertNotIn("ai_lab", body_of(answer))

    def test_a_body_that_is_not_json_goes_through(self):
        answer = explained(Passthrough(400, "application/json", iter([b"not json"]), {}))
        self.assertEqual(b"".join(answer.chunks), b"not json")

    def test_a_plain_string_error_is_read_too(self):
        body = json.dumps({"error": BY_PROMPT}).encode()
        answer = explained(Passthrough(400, "application/json", iter([body]), {}))
        self.assertEqual(body_of(answer)["ai_lab"]["prompt_tokens"], 40000)

    def test_success_and_other_statuses_are_not_read(self):
        chunks = iter([b"{}"])
        ok = Passthrough(200, "application/json", chunks, {})
        self.assertIs(explained(ok).chunks, chunks, "a success must stay a lazy stream")
        self.assertIs(explained(refusal(BY_TOKENS, status=500)).status, 500)
        text = refusal(BY_TOKENS, content_type="text/plain")
        self.assertIs(explained(text), text)


class ForwardTests(unittest.TestCase):
    def test_the_gateways_forward_explains_what_passthrough_returns(self):
        from unittest import mock

        import ai_lab.api.context_limit as module
        with mock.patch.object(module, "send", lambda *a, **k: refusal(BY_PROMPT)):
            answer = module.forward("http://x/v1/chat/completions", {}, module.Waits(),
                                    module.Hooks())
        self.assertEqual(body_of(answer)["ai_lab"]["context_size"], 32768)


class DescriptionTests(unittest.TestCase):
    def test_one_models_details_carry_the_live_pool(self):
        from ai_lab.api.routes.gateway import _one_model
        row = {"id": "q", "engine": "vllm", "loaded": True, "ready": True, "port": 8141,
               "shapes": [], "task": "text-generation", "capabilities": [],
               "params": {}, "live": {"kv_cache_tokens": 130389}}
        self.assertEqual(_one_model(row, detailed=True)["ai_lab"]["live"],
                         {"kv_cache_tokens": 130389})
        self.assertNotIn("live", _one_model(row)["ai_lab"],
                         "the listing stays short")
