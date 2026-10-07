import json
import unittest
from typing import Any

from ai_lab.providers.errors import ErrorKind, classify
from ai_lab.providers.prompt import JSON_ONLY, NO_TOOLS, completion, render


class ClassifyTests(unittest.TestCase):
    def test_the_three_classes(self):
        cases = [
            ("Claude AI usage limit reached|1760000000", ErrorKind.RATE_LIMIT, "usage limit"),
            ("ERROR: 429 Too Many Requests", ErrorKind.RATE_LIMIT, "429"),
            ("Invalid API key · Please run /login", ErrorKind.FATAL, "invalid api key"),
            ("error: unknown model gpt-9", ErrorKind.FATAL, "unknown model"),
            ("cannot start the CLI: [Errno 2] No such file", ErrorKind.FATAL,
             "cannot start the cli"),
        ]
        for text, kind, label in cases:
            with self.subTest(text=text):
                self.assertEqual(classify(text), (kind, label))

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
               {"messages": [{"role": "user", "content": None}]}]
        for payload in bad:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                render(payload)

    def test_the_answer_has_the_openai_shape(self):
        answer = completion("claude/sonnet", "Yes.")
        self.assertEqual(answer["object"], "chat.completion")
        self.assertEqual(answer["model"], "claude/sonnet")
        self.assertEqual(answer["choices"][0]["message"],
                         {"role": "assistant", "content": "Yes."})


if __name__ == "__main__":
    unittest.main()
