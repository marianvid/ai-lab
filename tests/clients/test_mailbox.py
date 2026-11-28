"""The mailbox between agents, read by the owner (2026-11-23)."""

import unittest

from ai_lab.clients.mailbox import Mailbox, check_client_name, party_of
from ai_lab.clients.security import SecuritySettings
from tests.support import temporary


class MailboxTests(unittest.TestCase):
    def setUp(self):
        self.box = Mailbox(temporary.folder(self) / "mailbox.jsonl", lambda: (3, 50))

    def test_a_party_is_a_client_named_mailbox_something(self):
        self.assertEqual(party_of("mailbox-sqwad"), "sqwad")
        self.assertIsNone(party_of("vm-sqwad"))
        self.assertIsNone(party_of("mailbox-"))
        self.assertIsNone(party_of(None))

    def test_owner_and_all_are_not_party_names(self):
        # A client named mailbox-owner must never read every message as the owner.
        self.assertIsNone(party_of("mailbox-owner"))
        self.assertIsNone(party_of("mailbox-all"))
        for name in ("mailbox-owner", "mailbox-all", "mailbox-Bad"):
            with self.assertRaises(ValueError):
                check_client_name(name)
        check_client_name("mailbox-sqwad")
        check_client_name("vm-sqwad")

    def test_messages_get_rising_ids_and_are_read_after_an_id(self):
        first = self.box.post("sqwad", "ai-lab", "hello")
        second = self.box.post("ai-lab", "sqwad", "hi")
        self.assertEqual((first["id"], second["id"]), (1, 2))
        self.assertEqual([m["id"] for m in self.box.read("sqwad", 1)["messages"]], [2])
        self.assertEqual(self.box.read("sqwad", 2), {"messages": [], "last": 2})

    def test_a_party_reads_what_is_to_it_to_all_or_from_it(self):
        self.box.post("sqwad", "ai-lab", "one")
        self.box.post("kiro", "ai-lab", "two")
        self.box.post("ai-lab", "all", "three")
        self.assertEqual([m["text"] for m in self.box.read("sqwad")["messages"]], ["one", "three"])
        self.assertEqual([m["text"] for m in self.box.read("ai-lab")["messages"]], ["one", "two", "three"])

    def test_the_owner_reads_everything(self):
        self.box.post("sqwad", "kiro", "private to kiro")
        self.assertEqual(len(self.box.read("owner")["messages"]), 1)

    def test_the_oldest_are_dropped_past_keep(self):
        for n in range(5):
            self.box.post("sqwad", "ai-lab", f"m{n}")
        self.assertEqual([m["text"] for m in self.box.read("owner")["messages"]], ["m2", "m3", "m4"])
        self.assertEqual(self.box.post("sqwad", "ai-lab", "m5")["id"], 6)

    def test_bad_messages_are_refused(self):
        with self.assertRaises(ValueError):
            self.box.post("sqwad", "ai-lab", "")
        with self.assertRaises(ValueError):
            self.box.post("sqwad", "ai-lab", "x" * 51)
        with self.assertRaises(ValueError):
            self.box.post("sqwad", "Bad Name!", "x")


class MailboxLimitsTests(unittest.TestCase):
    """How many messages are kept and how long one may be: Settings → Security values."""

    def setUp(self):
        self.security = SecuritySettings(temporary.folder(self) / "security.json")

    def test_the_defaults_and_their_change(self):
        self.assertEqual(self.security.mailbox_limits(), (2000, 20000))
        view = self.security.update({"mailbox_keep": 50, "mailbox_max_chars": 500})
        self.assertEqual((view["mailbox_keep"], view["mailbox_max_chars"]), (50, 500))
        self.assertEqual(self.security.mailbox_limits(), (50, 500))

    def test_a_limit_out_of_range_is_refused(self):
        with self.assertRaises(ValueError):
            self.security.update({"mailbox_keep": 1})
        with self.assertRaises(ValueError):
            self.security.update({"mailbox_max_chars": 10 ** 9})

    def test_the_mailbox_reads_the_limits_at_each_post(self):
        box = Mailbox(temporary.folder(self) / "m.jsonl", self.security.mailbox_limits)
        box.post("a", "b", "x" * 600)
        self.security.update({"mailbox_max_chars": 500})
        with self.assertRaises(ValueError):
            box.post("a", "b", "x" * 600)
