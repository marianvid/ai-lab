"""Caps on one client's calls: tokens in flight and calls at once (clients/caps.py)."""

import json
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from typing import Any, cast
from unittest import mock

import ai_lab.api.routes.gateway as gateway_routes
from ai_lab.api.passthrough import Passthrough
from ai_lab.api.router import Router
from ai_lab.api.routes import clients as client_routes
from ai_lab.api.server import Handler
from ai_lab.clients.access import Access
from ai_lab.clients.caps import (CallOverClientCapError, ClientCaps,
                                 OverClientCapError, answer_limit, prompt_estimate)
from ai_lab.clients.route_classes import model_api
from ai_lab.clients.sessions import COOKIE
from ai_lab.config.policy import CHARS_PER_TOKEN
from ai_lab.events import EventBus
from ai_lab.gateway.admission.replaced import Replacements
from ai_lab.gateway.limits import Patience
from tests.support import temporary

CHAT = "/v1/chat/completions"


def call(prompt_chars: int, max_tokens: int | None = 100) -> dict:
    body: dict = {"model": "qwen", "messages": [{"role": "user", "content": "x" * prompt_chars}]}
    if max_tokens is not None:
        body["max_tokens"] = max_tokens
    return body


class CapsTest(unittest.TestCase):
    def setUp(self):
        self.path = temporary.folder(self) / "client_caps.json"
        self.caps = ClientCaps(self.path)

    def test_a_client_without_caps_is_not_counted(self):
        give_back = self.caps.admit("scheduler-linux", CHAT, call(300_000), 4096)
        give_back()
        self.assertEqual(self.caps.view("scheduler-linux")["in_flight"], {"tokens": 0, "calls": 0})

    def test_a_call_counts_its_prompt_estimate_plus_its_answer_limit(self):
        self.caps.set("vm-sqwad", {"max_tokens_in_flight": 100_000})
        body = call(3000, 4096)
        self.caps.admit("vm-sqwad", CHAT, body, 512)
        held = self.caps.view("vm-sqwad")["in_flight"]
        self.assertEqual(held["calls"], 1)
        self.assertEqual(held["tokens"], prompt_estimate(body) + 4096)
        self.assertGreaterEqual(prompt_estimate(body), int(3000 / CHARS_PER_TOKEN))

    def test_no_answer_limit_counts_the_gateways_default_cap(self):
        self.assertEqual(answer_limit(call(10, None), 4096), 4096)
        self.assertEqual(answer_limit({"max_completion_tokens": 300}, 4096), 300)

    def test_over_the_token_cap_is_refused_at_once_with_a_retry_time(self):
        self.caps.set("vm-sqwad", {"max_tokens_in_flight": 10_000})
        self.caps.admit("vm-sqwad", CHAT, call(9000, 4000), 0)
        with self.assertRaises(OverClientCapError) as refused:
            self.caps.admit("vm-sqwad", CHAT, call(9000, 4000), 0)
        self.assertEqual(refused.exception.headers, {"Retry-After": "5"})
        facts = refused.exception.detail["ai_lab"]
        self.assertEqual(facts["reason"], "over_client_cap")
        self.assertEqual(facts["calls_in_flight"], 1)
        self.assertEqual(self.caps.view("vm-sqwad")["in_flight"]["calls"], 1,
                         "a refused call must not be counted")

    def test_over_the_calls_cap_is_refused(self):
        self.caps.set("vm-sqwad", {"max_calls_at_once": 2})
        first = self.caps.admit("vm-sqwad", CHAT, call(10), 0)
        self.caps.admit("vm-sqwad", CHAT, call(10), 0)
        with self.assertRaises(OverClientCapError):
            self.caps.admit("vm-sqwad", CHAT, call(10), 0)
        first()
        self.caps.admit("vm-sqwad", CHAT, call(10), 0)

    def test_giving_back_twice_counts_once(self):
        self.caps.set("vm-sqwad", {"max_calls_at_once": 4})
        give_back = self.caps.admit("vm-sqwad", CHAT, call(10), 0)
        self.caps.admit("vm-sqwad", CHAT, call(10), 0)
        give_back()
        give_back()
        self.assertEqual(self.caps.view("vm-sqwad")["in_flight"]["calls"], 1)

    def test_one_call_larger_than_the_whole_cap_is_a_bad_request(self):
        self.caps.set("vm-sqwad", {"max_tokens_in_flight": 5000})
        with self.assertRaises(CallOverClientCapError) as refused:
            self.caps.admit("vm-sqwad", CHAT, call(100, 8000), 0)
        self.assertEqual(refused.exception.detail["ai_lab"]["reason"], "call_over_client_cap")

    def test_other_paths_and_forms_pass_uncounted(self):
        self.caps.set("vm-sqwad", {"max_calls_at_once": 1})
        self.caps.admit("vm-sqwad", "/v1/embeddings", {"input": "x"}, 0)
        self.caps.admit("vm-sqwad", "/v1/audio/speech", {"input": "x"}, 0)
        self.assertEqual(self.caps.view("vm-sqwad")["in_flight"]["calls"], 0)

    def test_caps_are_kept_cleared_and_checked(self):
        self.caps.set("vm-sqwad", {"max_tokens_in_flight": 77141, "max_calls_at_once": 4})
        again = ClientCaps(self.path)
        self.assertEqual(again.of("vm-sqwad"), {"max_tokens_in_flight": 77141, "max_calls_at_once": 4})
        self.assertEqual(self.path.stat().st_mode & 0o777, 0o600)
        self.caps.set("vm-sqwad", {"max_calls_at_once": None})
        self.assertEqual(self.caps.of("vm-sqwad"), {"max_tokens_in_flight": 77141})
        self.caps.set("vm-sqwad", {"max_tokens_in_flight": 0})
        self.assertEqual(self.caps.of("vm-sqwad"), {})
        with self.assertRaises(ValueError):
            self.caps.set("vm-sqwad", {"max_calls_at_once": 0.5, "nonsense": 1})
        with self.assertRaises(ValueError):
            self.caps.set("vm-sqwad", {"max_calls_at_once": 5000})

    def test_calls_at_the_same_time_never_pass_the_cap(self):
        self.caps.set("vm-sqwad", {"max_calls_at_once": 3})
        admitted, refused = [], []

        def one():
            try:
                admitted.append(self.caps.admit("vm-sqwad", CHAT, call(10), 0))
            except OverClientCapError:
                refused.append(1)
        threads = [threading.Thread(target=one) for _ in range(20)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual((len(admitted), len(refused)), (3, 17))


ANSWERED = Passthrough(200, "application/json", iter([b"{}"]), {})


class Lease:
    def __init__(self):
        self.instance_id = "qwen"
        self.port = 9
        self.model_name = "qwen-real"
        self.drop: frozenset[str] = frozenset()
        self.text = True
        self.progress = None
        self.released = 0

    def release(self):
        self.released += 1


class Gateway:
    first_byte_s = 5.0
    between_bytes_s = 1.0
    max_upload_bytes = 0
    max_upload_pixels = 0
    max_upload_dimension = 0
    patience = Patience(default_max_tokens=0)

    def __init__(self, fail=False):
        self.leases = []
        self.fail = fail
        self.replacements = Replacements()

    def acquire(self, wanted, shape=None, settings=None, still_wanted=None):
        if self.fail:
            raise RuntimeError("no place")
        self.leases.append(Lease())
        return self.leases[-1]

    def timeouts_for(self, task):
        return self.first_byte_s, self.between_bytes_s


class ForwarderTest(unittest.TestCase):
    """The front door counts a capped client's call before it takes a place, and gives it back after."""

    def setUp(self):
        self.access = Access.in_folder(temporary.folder(self), log=lambda _line: None)
        self.token = self.access.register("vm-sqwad")["token"]
        self.access.set_caps("vm-sqwad", {"max_calls_at_once": 1})
        self.headers = {"Authorization": f"Bearer {self.token}"}

    def held(self):
        caps = self.access.caps
        assert caps is not None
        return caps.view("vm-sqwad")["in_flight"]["calls"]

    def test_the_answer_closing_gives_back_the_place_and_the_count(self):
        gateway = Gateway()
        closed = {}

        def fake_forward(url, payload, waits, hooks, **_):
            closed["hooks"] = hooks
            return ANSWERED

        with mock.patch.object(gateway_routes, "forward", fake_forward):
            handle = gateway_routes._forwarder(cast(Any, gateway), CHAT, access=self.access)
            handle(body=call(10), headers=self.headers)
            self.assertEqual(self.held(), 1)
            with self.assertRaises(OverClientCapError):
                handle(body=call(10), headers=self.headers)
            self.assertEqual(len(gateway.leases), 1, "a refused call must not ask for a place")
            closed["hooks"].closed()
        self.assertEqual((self.held(), gateway.leases[0].released), (0, 1))

    def test_no_place_gives_the_count_back(self):
        with self.assertRaises(RuntimeError):
            gateway_routes._forwarder(cast(Any, Gateway(fail=True)), CHAT, access=self.access)(
                body=call(10), headers=self.headers)
        self.assertEqual(self.held(), 0)

    def test_a_failed_send_gives_both_back(self):
        gateway = Gateway()

        def broken(*_args, **_kwargs):
            raise ConnectionRefusedError("engine gone")

        with mock.patch.object(gateway_routes, "forward", broken), \
                self.assertRaises(ConnectionRefusedError):
            gateway_routes._forwarder(cast(Any, gateway), CHAT, access=self.access)(
                body=call(10), headers=self.headers)
        self.assertEqual((self.held(), gateway.leases[0].released), (0, 1))

    def test_an_uncapped_client_is_never_counted(self):
        other = {"Authorization": f"Bearer {self.access.register('scheduler-linux')['token']}"}
        gateway = Gateway()
        with mock.patch.object(gateway_routes, "forward", lambda *a, **k: ANSWERED):
            handle = gateway_routes._forwarder(cast(Any, gateway), CHAT, access=self.access)
            for _ in range(5):
                handle(body=call(10), headers=other)
        self.assertEqual(len(gateway.leases), 5)


class CapRoutesTest(unittest.TestCase):
    """The owner's routes, and a refusal as a client sees it."""

    def setUp(self):
        self.access = Access.in_folder(temporary.folder(self), log=lambda _line: None)
        router = Router()
        client_routes.register(router, self.access)

        def busy(**_):
            raise OverClientCapError("vm-sqwad is at its cap", 5, client="vm-sqwad")
        router.add("POST", "/v1/busy", busy)
        handler = type("H", (Handler,), {"router": router, "bus": EventBus(), "access": self.access})
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()

    def send(self, method, path, body=None):
        """A call; a management route is called as the owner's page does (a page session)."""
        data = None if body is None else json.dumps(body).encode()
        headers = {"Content-Type": "application/json"}
        if not model_api(method, path):
            headers["Cookie"] = f"{COOKIE}={self.access.sessions.start(kept=False)}"
        request = urllib.request.Request(self.base + path, data=data, method=method,
                                         headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                return response.status, json.loads(response.read()), response.headers
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read()), error.headers

    def test_the_owner_sets_caps_and_the_list_shows_them(self):
        self.access.register("vm-sqwad")
        status, body, _ = self.send("PATCH", "/api/clients/vm-sqwad/caps",
                                    {"max_tokens_in_flight": 77141, "max_calls_at_once": 4})
        self.assertEqual(status, 200)
        self.assertEqual(body["caps"], {"max_tokens_in_flight": 77141, "max_calls_at_once": 4})
        _, listed, _ = self.send("GET", "/api/clients")
        row = listed["clients"][0]
        self.assertEqual((row["caps"]["max_calls_at_once"], row["in_flight"]), (4, {"tokens": 0, "calls": 0}))

    def test_caps_for_an_unknown_client_are_refused(self):
        status, _, _ = self.send("PATCH", "/api/clients/nobody/caps", {"max_calls_at_once": 4})
        self.assertEqual(status, 404)

    def test_removing_a_client_removes_its_caps(self):
        self.access.register("vm-sqwad")
        self.access.set_caps("vm-sqwad", {"max_calls_at_once": 4})
        self.send("DELETE", "/api/clients/vm-sqwad")
        caps = self.access.caps
        assert caps is not None
        self.assertEqual(caps.of("vm-sqwad"), {})

    def test_a_refusal_is_a_429_with_retry_after(self):
        status, body, headers = self.send("POST", "/v1/busy", {})
        self.assertEqual(status, 429)
        self.assertEqual(headers["Retry-After"], "5")
        self.assertEqual(body["error"]["code"], "over_client_cap")
        self.assertEqual(body["ai_lab"]["retry_after_s"], 5)


if __name__ == "__main__":
    unittest.main()
