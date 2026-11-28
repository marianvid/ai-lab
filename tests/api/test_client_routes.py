"""The access routes and the door, over a real socket."""

import json
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

from ai_lab.api.passthrough import Passthrough
from ai_lab.api.router import Router
from ai_lab.api.routes import clients as client_routes
from ai_lab.api.server import Handler
from ai_lab.clients.access import Access
from ai_lab.clients.errors import ModelReplacedError
from ai_lab.clients.route_classes import model_api
from ai_lab.clients.sessions import COOKIE
from ai_lab.events import EventBus
from tests.support import temporary


def require_token(access):
    """Tokens required, the only way that holds: a password set and login on (the read's m1)."""
    access.set_password("a long password", None)
    access.update_security({"login_required": True, "require_token": True})

class ClientRoutesTest(unittest.TestCase):
    def setUp(self):
        self.access = Access.in_folder(temporary.folder(self), log=lambda _line: None)
        router = Router()
        self.router = router
        client_routes.register(router, self.access)
        router.add("POST", "/v1/echo", lambda body, **_: body)
        handler = type("H", (Handler,), {"router": router, "bus": EventBus(),
                                         "access": self.access})
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()

    def call(self, method, path, body=None, headers=None):
        """A call; with no headers, a management route is called as the owner's page does."""
        data = None if body is None else json.dumps(body).encode()
        if headers is None and not model_api(method, path.split("?")[0]):
            # Management is the page's (a page session), whatever the token switch says.
            headers = {"Cookie": f"{COOKIE}={self.access.sessions.start(kept=False)}"}
        request = urllib.request.Request(self.base + path, data=data, method=method,
                                         headers={"Content-Type": "application/json",
                                                  **(headers or {})})
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                return response.status, json.loads(response.read()), response.headers
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read()), error.headers

    def test_health_is_open_and_says_on_or_off(self):
        require_token(self.access)
        self.access.power.switch(False)
        status, body, _ = self.call("GET", "/api/health")
        self.assertEqual((status, body), (200, {"ok": True, "on": False}))

    def test_a_refusal_carries_its_code_both_ways(self):
        require_token(self.access)
        status, body, _ = self.call("GET", "/api/clients", headers={})   # nothing at all
        self.assertEqual(status, 401)
        self.assertEqual(body["ai_lab"], {"reason": "token_missing"})
        status, body, _ = self.call("POST", "/v1/echo", {"a": 1},
                                    {"Authorization": "Bearer wrong"})
        self.assertEqual(status, 401)
        self.assertEqual(body["error"]["code"], "token_invalid")
        self.assertEqual(body["error"]["type"], "ai_lab")

    def test_register_then_call_and_heartbeat(self):
        _, created, _ = self.call("POST", "/api/clients", {"name": "scheduler-linux"})
        require_token(self.access)
        auth = {"Authorization": f"Bearer {created['token']}"}
        status, echoed, _ = self.call("POST", "/v1/echo", {"a": 1}, auth)
        self.assertEqual((status, echoed), (200, {"a": 1}))
        status, beat, _ = self.call("POST", "/api/clients/heartbeat", {}, auth)
        self.assertEqual(status, 200)
        self.assertTrue(beat["on"])
        status, _, _ = self.call("POST", "/api/clients/heartbeat", {"map": {"cards": []}}, auth)
        self.assertEqual(status, 200, "an older scheduler's map is accepted and ignored")
        # The clients are management: the page's, not a token's (radio-lab, 2026-11-27).
        status, refused, _ = self.call("GET", "/api/clients", headers=auth)
        self.assertEqual((status, refused["ai_lab"]["reason"]), (403, "management_needs_login"))
        _, _, headers = self.call("POST", "/api/login", {"password": "a long password"})
        page = {"Cookie": headers["Set-Cookie"].split(";")[0]}
        _, listed, _ = self.call("GET", "/api/clients", headers=page)
        self.assertIsNotNone(listed["clients"][0]["heartbeat_at"])
        self.assertNotIn("map", listed["clients"][0])
        status, _, _ = self.call("DELETE", "/api/clients/scheduler-linux", headers=auth)
        self.assertEqual(status, 403)
        status, _, _ = self.call("DELETE", "/api/clients/scheduler-linux", headers=page)
        self.assertEqual(status, 200)

    def test_login_sets_the_cookie_and_the_session_lets_the_page_in(self):
        self.call("POST", "/api/security/password", {"new": "a long password"})
        self.call("PATCH", "/api/security", {"login_required": True, "require_token": True})
        status, _, _ = self.call("POST", "/api/login", {"password": "wrong"})
        self.assertEqual(status, 403)
        status, _, headers = self.call("POST", "/api/login", {"password": "a long password"})
        self.assertEqual(status, 200)
        cookie = {"Cookie": headers["Set-Cookie"].split(";")[0]}
        status, session, _ = self.call("GET", "/api/session", headers=cookie)
        self.assertTrue(session["logged_in"])
        status, view, _ = self.call("GET", "/api/security", headers=cookie)
        self.assertEqual((status, view["login_required"]), (200, True))
        status, _, headers = self.call("POST", "/api/logout", headers=cookie)
        self.assertIn("Max-Age=0", headers["Set-Cookie"])

    def test_off_refuses_model_calls_with_the_reason_and_on_serves_again(self):
        status, power, _ = self.call("POST", "/api/power", {"on": False})
        self.assertEqual((status, power["on"]), (200, False))
        status, body, _ = self.call("POST", "/v1/echo", {"a": 1})
        self.assertEqual(status, 503)
        self.assertEqual(body["error"]["code"], "ai_lab_off")
        self.assertIn("since", body["ai_lab"])
        status, _, _ = self.call("GET", "/api/power")
        self.assertEqual(status, 200)
        self.call("POST", "/api/power", {"on": True})
        status, _, _ = self.call("POST", "/v1/echo", {"a": 1})
        self.assertEqual(status, 200)

    def test_a_stream_cut_by_ai_lab_ends_with_an_event_that_says_why(self):
        def chunks():
            yield b"data: one\n\n"
            raise ModelReplacedError("qwen was taken off the card for coder",
                                     model="qwen", replaced_by="coder")

        self.router.add("POST", "/v1/stream", lambda **_: Passthrough(
            status=200, content_type="text/event-stream", chunks=chunks(), headers={},
            close=lambda: None))
        request = urllib.request.Request(self.base + "/v1/stream", data=b"{}", method="POST")
        with urllib.request.urlopen(request, timeout=5) as response:
            body = response.read().decode()
        self.assertTrue(body.startswith("data: one\n\n"))
        last = json.loads(body.strip().split("\n\n")[-1][len("data: "):])
        self.assertEqual(last["error"]["code"], "model_replaced")
        self.assertEqual(last["ai_lab"]["replaced_by"], "coder")

    def test_the_page_opens_a_session_when_login_is_off(self):
        with urllib.request.urlopen(self.base + "/", timeout=5) as response:
            self.assertIn("HttpOnly", response.headers["Set-Cookie"])

    def test_the_mailbox_between_two_parties_and_the_owner(self):
        _, sqwad, _ = self.call("POST", "/api/clients", {"name": "mailbox-sqwad"})
        _, lab, _ = self.call("POST", "/api/clients", {"name": "mailbox-ai-lab"})
        as_sqwad = {"Authorization": f"Bearer {sqwad['token']}"}
        as_lab = {"Authorization": f"Bearer {lab['token']}"}
        status, sent, _ = self.call("POST", "/api/mailbox", {"to": "ai-lab", "text": "hello"}, as_sqwad)
        self.assertEqual((status, sent["from"], sent["to"]), (200, "sqwad", "ai-lab"))
        status, got, _ = self.call("GET", "/api/mailbox?after=0", headers=as_lab)
        self.assertEqual((status, [m["text"] for m in got["messages"]]), (200, ["hello"]))
        status, _, _ = self.call("POST", "/api/mailbox", {"to": "sqwad", "text": ""}, as_lab)
        self.assertEqual(status, 400)

    def test_a_mailbox_token_opens_the_mailbox_only(self):
        _, sqwad, _ = self.call("POST", "/api/clients", {"name": "mailbox-sqwad"})
        as_sqwad = {"Authorization": f"Bearer {sqwad['token']}"}
        status, body, _ = self.call("POST", "/v1/echo", {"a": 1}, as_sqwad)
        self.assertEqual((status, body["ai_lab"]["reason"]), (403, "access_denied"))
        status, _, _ = self.call("GET", "/api/clients", headers=as_sqwad)
        self.assertEqual(status, 403)

    def test_the_mailbox_refuses_a_model_token_and_no_token(self):
        _, model_client, _ = self.call("POST", "/api/clients", {"name": "vm-sqwad"})
        status, _, _ = self.call("GET", "/api/mailbox",
                                 headers={"Authorization": f"Bearer {model_client['token']}"})
        self.assertEqual(status, 403)
        status, _, _ = self.call("GET", "/api/mailbox")
        self.assertEqual(status, 403)

    def test_no_client_can_become_the_owner_of_the_mailbox(self):
        status, body, _ = self.call("POST", "/api/clients", {"name": "mailbox-owner"})
        self.assertEqual(status, 400)
        status, _, _ = self.call("POST", "/api/clients", {"name": "mailbox-all"})
        self.assertEqual(status, 400)

    def test_the_owner_reads_the_whole_mailbox_from_the_page(self):
        _, sqwad, _ = self.call("POST", "/api/clients", {"name": "mailbox-sqwad"})
        self.call("POST", "/api/mailbox", {"to": "kiro", "text": "x"},
                  {"Authorization": f"Bearer {sqwad['token']}"})
        with urllib.request.urlopen(self.base + "/", timeout=5) as response:
            cookie = {"Cookie": response.headers["Set-Cookie"].split(";")[0]}
        status, got, _ = self.call("GET", "/api/mailbox", headers=cookie)
        self.assertEqual((status, len(got["messages"])), (200, 1))
        status, sent, _ = self.call("POST", "/api/mailbox", {"to": "sqwad", "text": "from me"}, cookie)
        self.assertEqual((status, sent["from"]), (200, "owner"))

    def test_the_event_stream_is_behind_the_door(self):
        require_token(self.access)
        with self.assertRaises(urllib.error.HTTPError) as refused:
            urllib.request.urlopen(self.base + "/api/events", timeout=5)
        self.assertEqual(refused.exception.code, 401)


if __name__ == "__main__":
    unittest.main()
