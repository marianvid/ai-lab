"""When clients were last heard from; web sessions; Settings → Security."""

import tempfile
import unittest
from pathlib import Path

from ai_lab.clients.errors import AccessDeniedError
from ai_lab.clients.presence import Presence
from ai_lab.clients.security import SecuritySettings
from ai_lab.clients.sessions import COOKIE, SessionBook, cookie_header, digest, session_of


class Clock:
    def __init__(self, now=1000.0):
        self.now = now

    def __call__(self):
        return self.now


class PresenceTest(unittest.TestCase):
    def test_last_call_and_last_heartbeat_per_client(self):
        clock = Clock()
        presence = Presence(clock)
        presence.seen("voice-studio")
        clock.now = 1200.0
        self.assertEqual(presence.heartbeat("scheduler-linux"), 1200.0)
        rows = presence.view(["voice-studio", "scheduler-linux", "never"])
        self.assertEqual(rows[0], {"name": "voice-studio", "last_call_at": 1000.0,
                                   "heartbeat_at": None})
        self.assertEqual(rows[1], {"name": "scheduler-linux", "last_call_at": 1200.0,
                                   "heartbeat_at": 1200.0})
        self.assertEqual(rows[2]["last_call_at"], None)

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

    def test_a_session_survives_a_restart_and_only_its_digest_is_on_disk(self):
        path = Path(tempfile.mkdtemp()) / "sessions.json"
        clock = Clock()
        session = SessionBook(clock, lifetime_s=60, path=path).start()
        self.assertNotIn(session, path.read_text())
        self.assertIn(digest(session), path.read_text())
        again = SessionBook(clock, lifetime_s=60, path=path)
        self.assertTrue(again.valid(session))
        clock.now += 61
        self.assertFalse(SessionBook(clock, lifetime_s=60, path=path).valid(session))

    def test_ending_sessions_removes_them_from_disk(self):
        path = Path(tempfile.mkdtemp()) / "sessions.json"
        book = SessionBook(path=path)
        first, second = book.start(), book.start()
        book.end(first)
        self.assertFalse(SessionBook(path=path).valid(first))
        self.assertTrue(SessionBook(path=path).valid(second))
        book.end_all()
        self.assertFalse(SessionBook(path=path).valid(second))

    def test_an_unreadable_book_starts_empty(self):
        path = Path(tempfile.mkdtemp()) / "sessions.json"
        for text in ("{not json", "[1, 2]", '{"k": "soon"}'):
            path.write_text(text)
            with self.subTest(text=text):
                self.assertEqual(len(SessionBook(path=path)), 0)

    def test_the_lifetime_is_read_when_a_session_starts(self):
        days = [1.0]
        clock = Clock()
        book = SessionBook(clock, lifetime_s=lambda: days[0] * 100)
        first = book.start()
        days[0] = 3.0
        second = book.start()
        clock.now += 150
        self.assertEqual((book.valid(first), book.valid(second)), (False, True))

    def test_page_sessions_live_a_day_in_memory_and_are_capped(self):
        from unittest import mock
        path = Path(tempfile.mkdtemp()) / "sessions.json"
        clock = Clock()
        book = SessionBook(clock, path=path)
        with mock.patch("ai_lab.clients.sessions.PAGE_SESSIONS_MAX", 3):
            first, *rest = [book.start(kept=False) for _ in range(4)]
        self.assertFalse(book.valid(first), "the oldest page session stays past the cap")
        self.assertTrue(all(book.valid(item) for item in rest))
        self.assertFalse(path.exists())
        clock.now += 86400 + 1
        self.assertFalse(book.valid(rest[-1]))

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
                                           "password_set": False, "token_holds": True,
                                           "password_hash": {"kind": "scrypt", "n": 2 ** 14,
                                                             "r": 8, "p": 1},
                                           "mailbox_keep": 2000, "mailbox_max_chars": 20000,
                                           "session_days": 30, "password_min": 8})

    def test_the_shortest_password_is_a_setting_never_below_eight(self):
        # Audit m22: fixed in code before.
        security = SecuritySettings(self.path)
        with self.assertRaises(ValueError):
            security.update({"password_min": 7})
        security.update({"password_min": 12})
        with self.assertRaisesRegex(ValueError, "at least 12 characters"):
            security.set_password("eleven-char")
        security.set_password("twelve-chars")

    def test_a_machine_already_half_protected_is_not_changed_and_still_saves(self):
        # Both machines were in this state when m1 was built: nothing locks
        # them out and nothing rewrites their switches; the page warns.
        self.path.write_text('{"require_token": true}')
        security = SecuritySettings(self.path)
        self.assertFalse(security.token_holds)
        security.update({"mailbox_keep": 500, "session_days": 90})
        security.update({"require_token": True})
        self.assertTrue(security.require_token)
        self.assertEqual((security.view()["session_days"], security.session_s), (90, 90 * 86400.0))
        security.set_password("correct horse")
        security.update({"login_required": True})
        self.assertTrue(security.token_holds)

    def test_a_damaged_state_file_names_itself_and_is_not_replaced(self):
        # Audit L51: a JSON traceback stopped the start without naming the file.
        self.path.write_text('{"require_token": tr')
        with self.assertRaisesRegex(ValueError, "security.json cannot be read"):
            SecuritySettings(self.path)
        self.assertEqual(self.path.read_text(), '{"require_token": tr')

    def test_session_days_has_bounds(self):
        security = SecuritySettings(self.path)
        for days in (0, 366):
            with self.subTest(days=days), self.assertRaises(ValueError):
                security.update({"session_days": days})

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

    def test_a_token_required_with_login_off_is_refused_it_would_protect_nothing(self):
        # The line-by-line read's m1: with login off the page hands any visitor
        # a session admitted as `web`, so the token switch held nothing.
        security = SecuritySettings(self.path)
        with self.assertRaisesRegex(ValueError, "only holds with login on"):
            security.update({"require_token": True})
        security.set_password("correct horse")
        security.update({"login_required": True, "require_token": True})
        with self.assertRaisesRegex(ValueError, "only holds with login on"):
            security.update({"login_required": False})
        security.update({"require_token": False, "login_required": False})
        self.assertFalse(security.require_token or security.login_required)

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
