"""The access routes and the door, over a real socket."""

import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

from ai_lab.api.router import Router
from ai_lab.api.routes import clients as client_routes
from ai_lab.api.server import Handler
from ai_lab.clients.access import Access
from ai_lab.events import EventBus


class ClientRoutesTest(unittest.TestCase):
    def setUp(self):
        self.access = Access.in_folder(Path(tempfile.mkdtemp()), log=lambda _line: None)
        router = Router()
        client_routes.register(router, self.access, is_on=lambda: False)
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
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(self.base + path, data=data, method=method,
                                         headers={"Content-Type": "application/json",
                                                  **(headers or {})})
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                return response.status, json.loads(response.read()), response.headers
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read()), error.headers

    def test_health_is_open_and_says_on_or_off(self):
        self.access.update_security({"require_token": True})
        status, body, _ = self.call("GET", "/api/health")
        self.assertEqual((status, body), (200, {"ok": True, "on": False}))

    def test_a_refusal_carries_its_code_both_ways(self):
        self.access.update_security({"require_token": True})
        status, body, _ = self.call("GET", "/api/clients")
        self.assertEqual(status, 401)
        self.assertEqual(body["ai_lab"], {"reason": "token_missing"})
        status, body, _ = self.call("POST", "/v1/echo", {"a": 1},
                                    {"Authorization": "Bearer wrong"})
        self.assertEqual(status, 401)
        self.assertEqual(body["error"]["code"], "token_invalid")
        self.assertEqual(body["error"]["type"], "ai_lab")

    def test_register_then_call_and_heartbeat(self):
        _, created, _ = self.call("POST", "/api/clients", {"name": "scheduler-linux"})
        self.access.update_security({"require_token": True})
        auth = {"Authorization": f"Bearer {created['token']}"}
        status, echoed, _ = self.call("POST", "/v1/echo", {"a": 1}, auth)
        self.assertEqual((status, echoed), (200, {"a": 1}))
        status, beat, _ = self.call("POST", "/api/clients/heartbeat", {"map": {"cards": []}}, auth)
        self.assertEqual(status, 200)
        self.assertFalse(beat["on"])
        _, listed, _ = self.call("GET", "/api/clients", headers=auth)
        self.assertEqual(listed["clients"][0]["map"], {"cards": []})
        status, _, _ = self.call("DELETE", "/api/clients/scheduler-linux", headers=auth)
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

    def test_the_page_opens_a_session_when_login_is_off(self):
        with urllib.request.urlopen(self.base + "/", timeout=5) as response:
            self.assertIn("HttpOnly", response.headers["Set-Cookie"])

    def test_the_event_stream_is_behind_the_door(self):
        self.access.update_security({"require_token": True})
        with self.assertRaises(urllib.error.HTTPError) as refused:
            urllib.request.urlopen(self.base + "/api/events", timeout=5)
        self.assertEqual(refused.exception.code, 401)


if __name__ == "__main__":
    unittest.main()
