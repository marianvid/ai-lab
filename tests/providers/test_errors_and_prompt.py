import json
import unittest
from typing import Any

from ai_lab.providers.errors import SAID_LENGTH, ErrorKind, classify, said
from ai_lab.providers.prompt import (
    JSON_ONLY,
    NO_TOOLS,
    completion,
    render,
    unfence,
    wants_json,
)


class ClassifyTests(unittest.TestCase):
    def test_the_three_classes(self):
        cases = [
            ("Claude AI usage limit reached|1760000000", ErrorKind.RATE_LIMIT, "usage limit"),
            ("ERROR: 429 Too Many Requests", ErrorKind.RATE_LIMIT, "429"),
            ("Invalid API key", ErrorKind.FATAL, "invalid api key"),
            ("401 Unauthorized", ErrorKind.FATAL, "unauthorized"),
            ("error: unknown model gpt-9", ErrorKind.FATAL, "unknown model"),
            ("cannot start the CLI: [Errno 2] No such file", ErrorKind.FATAL,
             "cannot start the cli"),
            ('{"type":"invalid_request_error","message":"The \'gpt-5.6-sol\' model is not '
             'supported when using Codex with a ChatGPT account."}', ErrorKind.FATAL,
             "is not supported"),
        ]
        for text, kind, label in cases:
            with self.subTest(text=text):
                self.assertEqual(classify(text), (kind, label))

    def test_a_number_or_a_word_inside_other_text_is_not_a_rate_limit(self):
        # Audit L40: markers are whole words, and 429 only as a status.
        for text in ("request id req_4291 failed", "wrote 429 bytes then the pipe broke",
                     "cannot open /tmp/capacityplan.txt", "quotas.json: no such file"):
            with self.subTest(text=text):
                self.assertIsNot(classify(text)[0], ErrorKind.RATE_LIMIT)
        for text in ("HTTP 429", "API Error: 429 {}", "status: 429", "429 Too Many Requests",
                     "the model is at capacity", "quota exceeded"):
            with self.subTest(text=text):
                self.assertIs(classify(text)[0], ErrorKind.RATE_LIMIT)

    def test_a_dead_login_in_each_vendor_s_own_words(self):
        # Each text as the CLI printed it (see LOGIN_MARKERS for where and when).
        seen = [
            "Failed to authenticate: OAuth session expired and could not be refreshed",
            "Failed to authenticate. API Error: 401 Invalid bearer token",
            "Not logged in · Please run /login",
            "Invalid API key · Please run /login",
            "Login expired · Please run /login",
            "socket: failed to connect to websocket: HTTP error: 401 Unauthorized | "
            "ERROR: unexpected status 401 Unauthorized: Missing bearer or basic "
            "authentication in header, url: https://api.openai.com/v1/responses",
        ]
        for text in seen:
            with self.subTest(text=text):
                self.assertEqual(classify(text), (ErrorKind.LOGIN, "login expired"))

    def test_nothing_else_is_taken_for_a_dead_login(self):
        # A false "expired" sends every caller to its fallback: these stay what they are.
        for text, kind in (("timeout", ErrorKind.TRANSIENT),
                           ("Claude AI usage limit reached|1760000000", ErrorKind.RATE_LIMIT),
                           ("429 Too Many Requests", ErrorKind.RATE_LIMIT),
                           ("401 Unauthorized", ErrorKind.FATAL),
                           ("error: unknown model gpt-9", ErrorKind.FATAL),
                           ("error sending request for url", ErrorKind.TRANSIENT)):
            with self.subTest(text=text):
                self.assertEqual(classify(text)[0], kind)

    def test_what_a_dead_login_said_is_one_short_line_without_secrets(self):
        text = "line one\n  Failed to authenticate\nsk-made-up-value-for-the-test " \
               "req_12bb4239c16e4519977a68b63b5305db"
        words = said(text)
        self.assertEqual(words, "line one Failed to authenticate […] […]")
        self.assertLessEqual(len(said("x " * 500)), SAID_LENGTH)
        self.assertEqual(said(""), "no output")

    def test_unrecognised_text_is_transient_and_keeps_a_stable_label(self):
        first = classify("starting\nerror sending request for url (https://x/1234)")
        second = classify("error sending request for url (https://x/9876)")
        self.assertEqual(first[0], ErrorKind.TRANSIENT)
        self.assertEqual(first, second)
        self.assertEqual(classify("   "), (ErrorKind.TRANSIENT, "empty output"))


class RenderTests(unittest.TestCase):
    def test_one_user_message_is_the_prompt_after_the_instructions(self):
        text = render({"messages": [{"role": "system", "content": "Be brief."},
                                    {"role": "user", "content": "Hello?"}]})
        self.assertEqual(text, f"Be brief.\n\n{NO_TOOLS}\n\nHello?")

    def test_a_conversation_keeps_its_roles_and_text_parts(self):
        text = render({"messages": [
            {"role": "user", "content": [{"type": "text", "text": "Hi"}]},
            {"role": "assistant", "content": "Hello"},
            {"role": "user", "content": "Again"}]})
        self.assertTrue(text.endswith("User: Hi\n\nAssistant: Hello\n\nUser: Again"))

    def test_json_requests_become_instructions(self):
        schema = {"type": "object"}
        text = render({"messages": [{"role": "user", "content": "x"}],
                       "response_format": {"type": "json_schema",
                                           "json_schema": {"schema": schema}}})
        self.assertIn(JSON_ONLY, text)
        self.assertIn(json.dumps(schema), text)
        self.assertIn(JSON_ONLY, render({"messages": [{"role": "user", "content": "x"}],
                                         "response_format": {"type": "json_object"}}))

    def test_unusable_requests_are_refused(self):
        bad: list[Any] = [{}, {"messages": [{"role": "system", "content": "only rules"}]},
               {"messages": [{"role": "user", "content": [{"type": "image_url"}]}]},
               {"messages": [{"role": "user", "content": None}]},
               {"messages": ["just a string"]}]
        for payload in bad:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                render(payload)

    def test_the_answer_has_the_openai_shape(self):
        answer = completion("claude/sonnet", "Yes.")
        self.assertEqual(answer["object"], "chat.completion")
        self.assertEqual(answer["model"], "claude/sonnet")
        self.assertEqual(answer["choices"][0]["message"],
                         {"role": "assistant", "content": "Yes."})


class UnfenceTests(unittest.TestCase):
    def test_one_fenced_block_gives_its_inside(self):
        cases = {
            '```json\n{"answers": [1]}\n```': '{"answers": [1]}',
            '```\n[1, 2]\n```': "[1, 2]",
            '  ```JSON \n{"a": 1}\n\n```\n': '{"a": 1}',
            '```json\n{"a": 1}```': '{"a": 1}',
        }
        for fenced, plain in cases.items():
            with self.subTest(fenced=fenced):
                self.assertEqual(unfence(fenced), plain)

    def test_anything_else_is_left_as_it_is(self):
        for text in ('{"a": 1}', 'Here it is:\n```json\n{"a": 1}\n```',
                     '```json\n{"a": 1}\n```\nDone.', '```json\n{"a": "```"}\n```',
                     "```json {}```"):
            with self.subTest(text=text):
                self.assertEqual(unfence(text), text)

    def test_only_a_json_response_format_asks_for_json(self):
        self.assertTrue(wants_json({"response_format": {"type": "json_object"}}))
        self.assertTrue(wants_json({"response_format": {"type": "json_schema"}}))
        self.assertFalse(wants_json({"response_format": {"type": "text"}}))
        self.assertFalse(wants_json({}))


if __name__ == "__main__":
    unittest.main()
