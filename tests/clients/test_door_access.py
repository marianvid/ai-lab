"""The door: tokens, the web session, the open routes, log mode and require mode."""

import tempfile
import unittest
from pathlib import Path

from ai_lab.clients.access import Access
from ai_lab.clients.door import ANONYMOUS, WEB, bearer, route_key
from ai_lab.clients.errors import AccessDeniedError, TokenInvalidError, TokenMissingError
from ai_lab.clients.sessions import COOKIE


def cookie(header: str | None) -> dict:
    assert header is not None
    return {"Cookie": header.split(";")[0]}


class DoorTest(unittest.TestCase):
    def setUp(self):
        self.lines: list[str] = []
        self.access = Access.in_folder(Path(tempfile.mkdtemp()), log=self.lines.append)
        self.token = self.access.register("scheduler-linux")["token"]
        self.door = self.access.door

    def test_a_valid_token_names_its_client_and_is_seen(self):
        caller = self.door.admit("POST", "/v1/chat/completions",
                                 {"Authorization": f"Bearer {self.token}"})
        self.assertEqual(caller, "scheduler-linux")
        row = self.access.clients()["clients"][0]
        self.assertIsNotNone(row["last_call_at"])

    def test_log_mode_serves_and_logs_once_per_route(self):
        for _ in range(3):
            self.assertEqual(self.door.admit("GET", "/v1/models/x", {}), ANONYMOUS)
        self.door.admit("GET", "/v1/models", {"Authorization": "Bearer wrong"})
        self.assertEqual(len(self.lines), 2)
        self.assertIn("/v1/models", self.lines[0])
        self.door.admit("GET", "/v1/models", {}, "192.168.50.143")
        self.door.admit("GET", "/v1/models", {}, "192.168.50.143")
        self.assertEqual(len(self.lines), 3, "a new address is a new line, once")
        self.assertIn("from 192.168.50.143", self.lines[2])

    def test_require_mode_refuses_with_the_code(self):
        self.access.update_security({"require_token": True})
        with self.assertRaises(TokenMissingError) as missing:
            self.door.admit("GET", "/v1/models", {})
        self.assertEqual(missing.exception.detail, {"ai_lab": {"reason": "token_missing"}})
        with self.assertRaises(TokenInvalidError):
            self.door.admit("GET", "/v1/models", {"Authorization": "Bearer nope"})

    def test_the_open_routes_need_nothing(self):
        self.access.update_security({"require_token": True})
        for method, path in (("GET", "/api/health"), ("GET", "/api/session"),
                             ("POST", "/api/login")):
            self.assertEqual(self.door.admit(method, path, {}), ANONYMOUS)

    def test_the_web_session_is_let_in(self):
        self.access.update_security({"require_token": True})
        header = self.access.page_cookie({})
        self.assertEqual(self.door.admit("GET", "/api/instances", cookie(header)), WEB)

    def test_helpers(self):
        self.assertEqual(bearer({"Authorization": "bearer abc "}), "abc")
        self.assertIsNone(bearer({"Authorization": "Basic abc"}))
        self.assertIsNone(bearer({"Authorization": "Bearer "}))
        self.assertIsNone(bearer(None))
        self.assertEqual(route_key("/api/clients/x/y"), "/api/clients")


class AccessTest(unittest.TestCase):
    def setUp(self):
        self.access = Access.in_folder(Path(tempfile.mkdtemp()), log=lambda _line: None)

    def test_the_heartbeat_is_noted_and_a_map_is_not_kept(self):
        token = self.access.register("scheduler-mac")["token"]
        self.access.heartbeat({"Authorization": f"Bearer {token}"})
        row = self.access.clients()["clients"][0]
        self.assertIsNotNone(row["heartbeat_at"])
        self.assertNotIn("map", row)
        self.assertNotIn("down", row)
        with self.assertRaises(AccessDeniedError):
            self.access.heartbeat({})

    def test_login_logout_and_the_page_cookie(self):
        self.access.set_password("a long password", None)
        self.access.update_security({"login_required": True})
        self.assertIsNone(self.access.page_cookie({}))
        with self.assertRaises(AccessDeniedError):
            self.access.login("wrong")
        header = self.access.login("a long password")
        self.assertTrue(self.access.session(cookie(header))["logged_in"])
        cleared = self.access.logout(cookie(header))
        self.assertIn("Max-Age=0", cleared)
        self.assertFalse(self.access.session(cookie(header))["logged_in"])

    def test_with_login_off_the_page_opens_its_session_once(self):
        header = self.access.page_cookie({})
        self.assertTrue(cookie(header)["Cookie"].startswith(COOKIE))
        self.assertIsNone(self.access.page_cookie(cookie(header)))

    def test_switching_login_on_or_changing_the_password_ends_sessions(self):
        header = self.access.page_cookie({})
        self.access.set_password("a long password", None)
        self.assertTrue(self.access.session(cookie(header))["logged_in"])
        self.access.update_security({"login_required": True})
        self.assertFalse(self.access.session(cookie(header))["logged_in"])
        header = self.access.login("a long password")
        self.access.set_password("another password", "a long password")
        self.assertFalse(self.access.session(cookie(header))["logged_in"])

    def test_remove(self):
        self.access.register("voice-studio")
        self.access.remove("voice-studio")
        self.assertEqual(self.access.clients(), {"clients": []})


if __name__ == "__main__":
    unittest.main()
