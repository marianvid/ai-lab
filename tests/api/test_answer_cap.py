"""The default answer cap: added when a text request names none, dropped again when it causes a refusal."""

import json
import unittest

from ai_lab.api.answer_cap import HEADER, PlaceKeeper, add_cap, send_capped
from ai_lab.api.passthrough import Passthrough

TOO_LONG = {"error": {"message": "too long"},
            "ai_lab": {"reason": "context_too_long", "context_size": 32768,
                       "prompt_tokens": 30000, "max_output_tokens": 4096}}


class Released:
    """A stand-in for `Lease.release`: gives the place back once, as the real one does."""

    def __init__(self):
        self.count = 0

    def __call__(self):
        self.count = 1


def answer(status=200, body=b"{}", on_close=None) -> Passthrough:
    """An answer as `forward` returns it; `close` tells the hooks, as the real one does."""
    return Passthrough(status, "application/json", iter([body]), {},
                       on_close or (lambda: None))


class AddCapTests(unittest.TestCase):
    def test_a_request_without_a_limit_gets_the_cap(self):
        request: dict = {"messages": []}
        self.assertEqual(add_cap(request, "/v1/chat/completions", 4096), 4096)
        self.assertEqual(request["max_tokens"], 4096)

    def test_a_request_with_its_own_limit_keeps_it(self):
        for field in ("max_tokens", "max_completion_tokens"):
            request: dict = {field: 300}
            self.assertEqual(add_cap(request, "/v1/completions", 4096), 0)
            self.assertEqual(request, {field: 300})

    def test_a_cap_of_zero_adds_nothing(self):
        request: dict = {}
        self.assertEqual(add_cap(request, "/v1/chat/completions", 0), 0)
        self.assertEqual(request, {})

    def test_other_paths_are_left_alone(self):
        for path in ("/v1/messages", "/tokenize", "/v1/audio/transcriptions"):
            request: dict = {}
            self.assertEqual(add_cap(request, path, 4096), 0, path)
            self.assertEqual(request, {})


class SendCappedTests(unittest.TestCase):
    def test_the_cap_is_sent_and_said_in_a_header(self):
        released = Released()
        keeper = PlaceKeeper(released)
        sent = []

        def send(body):
            sent.append(dict(body))
            return answer(on_close=keeper)

        result = send_capped(send, {"messages": []}, "/v1/chat/completions", 4096, keeper)
        self.assertEqual(sent, [{"messages": [], "max_tokens": 4096}])
        self.assertEqual(result.headers[HEADER], "4096")
        self.assertEqual(released.count, 0, "the place is held until the answer closes")
        result.close()
        self.assertEqual(released.count, 1)

    def test_a_refusal_caused_by_the_cap_is_sent_again_without_it(self):
        released = Released()
        keeper = PlaceKeeper(released)
        sent = []

        def send(body):
            sent.append(dict(body))
            if "max_tokens" in body:
                refusal = answer(400, json.dumps(TOO_LONG).encode(), on_close=keeper)
                # `explained` reads a refusal whole, which closes it at once.
                refusal.close()
                return refusal
            return answer(on_close=keeper)

        result = send_capped(send, {"messages": []}, "/v1/chat/completions", 4096, keeper)
        self.assertEqual(sent, [{"messages": [], "max_tokens": 4096}, {"messages": []}])
        self.assertEqual(released.count, 0, "the place must survive the retry")
        self.assertNotIn(HEADER, result.headers)
        result.close()
        self.assertEqual(released.count, 1)

    def test_any_other_refusal_comes_back_as_it_was(self):
        released = Released()
        keeper = PlaceKeeper(released)
        body = json.dumps({"error": {"message": "bad temperature"}}).encode()

        def send(_body):
            refusal = answer(400, body, on_close=keeper)
            refusal.close()
            return refusal

        result = send_capped(send, {}, "/v1/chat/completions", 4096, keeper)
        self.assertEqual(result.status, 400)
        self.assertEqual(b"".join(result.chunks), body)
        result.close()
        self.assertEqual(released.count, 1, "closed while held, given back at the end")

    def test_a_body_that_is_not_json_is_not_a_length_refusal(self):
        keeper = PlaceKeeper(Released())
        result = send_capped(lambda _body: answer(400, b"nope"), {}, "/v1/completions",
                             4096, keeper)
        self.assertEqual(b"".join(result.chunks), b"nope")

    def test_without_a_cap_the_request_goes_once_as_it_is(self):
        sent = []

        def send(body):
            sent.append(dict(body))
            return answer()

        send_capped(send, {"max_tokens": 10}, "/v1/chat/completions", 4096,
                    PlaceKeeper(Released()))
        self.assertEqual(sent, [{"max_tokens": 10}])

    def test_a_failure_while_held_leaves_the_keeper_ready_to_give_back(self):
        released = Released()
        keeper = PlaceKeeper(released)

        def send(_body):
            raise TimeoutError

        with self.assertRaises(TimeoutError):
            send_capped(send, {}, "/v1/chat/completions", 4096, keeper)
        keeper()
        self.assertEqual(released.count, 1)


if __name__ == "__main__":
    unittest.main()
