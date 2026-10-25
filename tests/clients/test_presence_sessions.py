"""When clients were last heard from; web sessions; Settings → Security."""

import tempfile
import unittest
from pathlib import Path

from ai_lab.clients.errors import AccessDeniedError
from ai_lab.clients.presence import Presence
from ai_lab.clients.security import SecuritySettings
from ai_lab.clients.sessions import COOKIE, SessionBook, cookie_header, session_of


class Clock:
    def __init__(self, now=1000.0):
        self.now = now

    def __call__(self):
        return self.now


class PresenceTest(unittest.TestCase):
    def test_a_quiet_client_without_heartbeat_is_never_down(self):
        clock = Clock()
        presence = Presence(clock, down_after_s=180)
        presence.seen("voice-studio")
        clock.now += 10_000
        row = presence.view(["voice-studio"])[0]
        self.assertEqual(row["last_call_at"], 1000.0)
        self.assertFalse(row["down"])
        self.assertIsNone(row["map"])

    def test_a_scheduler_silent_too_long_is_down(self):
        clock = Clock()
        presence = Presence(clock, down_after_s=180)
        self.assertEqual(presence.heartbeat("scheduler-linux", {"cards": []}), 1000.0)
        clock.now += 179
        self.assertFalse(presence.view(["scheduler-linux"])[0]["down"])
        clock.now += 2
        row = presence.view(["scheduler-linux"])[0]
        self.assertTrue(row["down"])
        self.assertEqual(row["map"], {"cards": []})

    def test_a_heartbeat_needs_a_map(self):
        with self.assertRaises(ValueError):
            Presence().heartbeat("scheduler-linux", [])

    def test_an_unknown_name_has_an_empty_row(self):
        row = Presence().view(["never"])[0]
        self.assertIsNone(row["last_call_at"])


class SessionsTest(unittest.TestCase):
    def test_a_session_expires(self):
        clock = Clock()
        book = SessionBook(clock, lifetime_s=60)
        session = book.start()
        self.assertTrue(book.valid(session))
        clock.now += 61
        self.assertFalse(book.valid(session))
        self.assertFalse(book.valid(None))

    def test_end_and_end_all(self):
        book = SessionBook()
        first, second = book.start(), book.start()
        book.end(first)
        self.assertFalse(book.valid(first))
        self.assertTrue(book.valid(second))
        book.end_all()
        self.assertFalse(book.valid(second))

    def test_the_cookie_is_http_only_and_same_site(self):
        header = cookie_header("abc")
        self.assertIn("HttpOnly", header)
        self.assertIn("SameSite=Strict", header)
        self.assertEqual(session_of({"Cookie": f"other=1; {COOKIE}=abc"}), "abc")
        self.assertIsNone(session_of({"Cookie": "other=1"}))
        self.assertIsNone(session_of({}))
        self.assertIsNone(session_of({"Cookie": "\x00bad=\x01;;"}))


class SecurityTest(unittest.TestCase):
    def setUp(self):
        self.path = Path(tempfile.mkdtemp()) / "security.json"

    def test_defaults_let_everything_in(self):
        security = SecuritySettings(self.path)
        self.assertEqual(security.view(), {"require_token": False, "login_required": False,
                                           "password_set": False})

    def test_login_cannot_be_required_without_a_password(self):
        with self.assertRaises(ValueError):
            SecuritySettings(self.path).update({"login_required": True})

    def test_the_password_is_hashed_and_checked(self):
        security = SecuritySettings(self.path)
        security.set_password("correct horse")
        self.assertTrue(security.check("correct horse"))
        self.assertFalse(security.check("wrong one"))
        self.assertNotIn("correct horse", self.path.read_text())
        security.update({"login_required": True, "require_token": True})
        again = SecuritySettings(self.path)
        self.assertTrue(again.login_required and again.require_token)

    def test_changing_the_password_needs_the_current_one(self):
        security = SecuritySettings(self.path)
        security.set_password("first password")
        with self.assertRaises(AccessDeniedError):
            security.set_password("second password", "nope")
        security.set_password("second password", "first password")
        self.assertTrue(security.check("second password"))

    def test_a_short_password_and_an_unknown_setting_are_refused(self):
        security = SecuritySettings(self.path)
        with self.assertRaises(ValueError):
            security.set_password("short")
        with self.assertRaises(ValueError):
            security.update({"open_everything": True})
        self.assertFalse(security.check("anything"))


if __name__ == "__main__":
    unittest.main()
