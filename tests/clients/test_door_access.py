"""The door: tokens, the web session, the open routes, log mode and require mode."""

import json
import tempfile
import unittest
from pathlib import Path

from ai_lab.clients.access import Access
from ai_lab.clients.door import ANONYMOUS, WEB, bearer, route_key
from ai_lab.clients.errors import (AccessDeniedError, ManagementNeedsLoginError, TokenInvalidError,
                                   TokenMissingError)
from ai_lab.clients.sessions import COOKIE


def cookie(header: str | None) -> dict:
    assert header is not None
    return {"Cookie": header.split(";")[0]}



def require_token(access):
    """Tokens required, the only way that holds: a password set and login on (the read's m1)."""
    access.set_password("a long password", None)
    access.update_security({"login_required": True, "require_token": True})

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

    def test_a_token_reaches_the_model_api_and_not_management(self):
        # radio-lab, 2026-11-27: with login on, a token must not change Security.
        auth = {"Authorization": f"Bearer {self.token}"}
        for method, path in (("GET", "/v1/models"), ("POST", "/tokenize"),
                             ("POST", "/api/clients/heartbeat"), ("GET", "/api/gateway")):
            self.assertEqual(self.door.admit(method, path, auth), "scheduler-linux")
        for method, path in (("PATCH", "/api/security"), ("GET", "/api/clients"),
                             ("POST", "/api/instances/qwen/load")):
            with self.subTest(path), self.assertRaises(ManagementNeedsLoginError) as refused:
                self.door.admit(method, path, auth)
            self.assertEqual(refused.exception.detail["ai_lab"]["reason"], "management_needs_login")

    def test_the_page_session_reaches_management_with_login_off_and_on(self):
        page = {"Cookie": f"{COOKIE}={self.access.sessions.start(kept=False)}"}
        self.assertEqual(self.door.admit("PATCH", "/api/security", page), WEB)
        require_token(self.access)
        page = cookie(self.access.login("a long password"))
        self.assertEqual(self.door.admit("PATCH", "/api/security", page), WEB)

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
        require_token(self.access)
        with self.assertRaises(TokenMissingError) as missing:
            self.door.admit("GET", "/v1/models", {})
        self.assertEqual(missing.exception.detail, {"ai_lab": {"reason": "token_missing"}})
        with self.assertRaises(TokenInvalidError):
            self.door.admit("GET", "/v1/models", {"Authorization": "Bearer nope"})

    def test_the_open_routes_need_nothing(self):
        require_token(self.access)
        for method, path in (("GET", "/api/health"), ("GET", "/api/session"),
                             ("POST", "/api/login")):
            self.assertEqual(self.door.admit(method, path, {}), ANONYMOUS)

    def test_while_the_token_does_not_hold_a_page_session_call_is_said_once(self):
        # Before login goes on, the log names who gets in by a page cookie.
        folder = Path(tempfile.mkdtemp())
        (folder / "security.json").write_text('{"require_token": true}')
        lines: list[str] = []
        access = Access.in_folder(folder, log=lines.append)
        page = cookie(access.page_cookie({}))
        for _ in range(2):
            self.assertEqual(access.door.admit("GET", "/api/gateway", page, "console-host"), WEB)
        said = [line for line in lines if "page session" in line]
        self.assertEqual(len(said), 1)
        self.assertIn("/api/gateway from console-host", said[0])
        with self.assertRaises(TokenMissingError):
            access.door.admit("GET", "/api/gateway", {}, "console-host")

    def test_the_web_session_is_let_in(self):
        require_token(self.access)
        header = self.access.login("a long password")
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

    def test_a_login_holds_through_a_restart_and_its_cookie_for_the_days_set(self):
        # The owner's "remember me": closing the browser keeps the cookie
        # (Max-Age), and a restart (a new Access on the same folder) keeps it.
        folder = Path(tempfile.mkdtemp())
        access = Access.in_folder(folder, log=lambda _line: None)
        access.set_password("a long password", None)
        access.update_security({"login_required": True, "session_days": 7})
        header = access.login("a long password")
        self.assertIn(f"Max-Age={7 * 86400}", header)
        again = Access.in_folder(folder, log=lambda _line: None)
        self.assertTrue(again.session(cookie(header))["logged_in"])
        again.logout(cookie(header))
        self.assertFalse(Access.in_folder(folder, log=lambda _line: None)
                         .session(cookie(header))["logged_in"])

    def test_page_sessions_with_login_off_are_never_written_to_disk(self):
        # The reviewer's FIX FIRST: every cookie-less GET / wrote a 30-day
        # session; a check a minute grew the file for a month.
        folder = Path(tempfile.mkdtemp())
        access = Access.in_folder(folder, log=lambda _line: None)
        pages = [access.page_cookie({}) for _ in range(100)]
        self.assertFalse((folder / "sessions.json").exists())
        self.assertIn("Max-Age=86400", str(pages[0]))
        self.assertTrue(access.session(cookie(pages[-1]))["logged_in"])
        access.set_password("a long password", None)
        access.update_security({"login_required": True})
        self.assertFalse(access.session(cookie(pages[-1]))["logged_in"])
        header = access.login("a long password")
        written = (folder / "sessions.json").read_text()
        self.assertEqual(len(json.loads(written)), 1)
        self.assertTrue(Access.in_folder(folder, log=lambda _line: None)
                        .session(cookie(header))["logged_in"])

    def test_the_session_view_says_whether_the_token_holds(self):
        self.assertTrue(self.access.session({})["token_holds"])
        (self.access.security.path).write_text('{"require_token": true}')
        half = Access.in_folder(self.access.security.path.parent, log=lambda _line: None)
        self.assertFalse(half.session({})["token_holds"])

    def test_remove(self):
        self.access.register("voice-studio")
        self.access.remove("voice-studio")
        self.assertEqual(self.access.clients(), {"clients": []})


if __name__ == "__main__":
    unittest.main()
