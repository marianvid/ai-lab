"""The host helper (runs on the Proxmox host): which kernel lines it keeps, and how it writes them."""

import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from ai_lab.gpu_health.kernel import host_helper as helper


def journal(message, stamp="1761300000000000"):
    return json.dumps({"MESSAGE": message, "__REALTIME_TIMESTAMP": stamp})


class Lines(unittest.TestCase):
    def test_xid_and_aer_are_kept_the_rest_is_not(self):
        entry = helper.kernel_entry(journal("NVRM: Xid (PCI:0000:01:00): 79, fallen off the bus"))
        self.assertEqual(entry, {"ts": 1761300000.0, "kind": "kernel",
                                 "line": "NVRM: Xid (PCI:0000:01:00): 79, fallen off the bus"})
        self.assertIsNotNone(helper.kernel_entry(journal("pcieport 0000:00:01.1: AER: Corrected")))
        self.assertIsNone(helper.kernel_entry(journal("usb 1-1: new device")))
        self.assertIsNone(helper.kernel_entry("not json"))
        self.assertIsNone(helper.kernel_entry("[1]"))

    def test_a_message_that_was_not_text(self):
        raw = json.dumps({"MESSAGE": list(b"NVRM: Xid (PCI:0000:01:00): 13, \xff")})
        entry = helper.kernel_entry(raw)
        assert entry is not None
        self.assertTrue(entry["line"].startswith("NVRM: Xid"))
        self.assertGreater(entry["ts"], 0)


class Files(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.folder = Path(folder.name) / "helper"

    def test_written_readable_and_rotated(self):
        output = helper.Output(self.folder)
        output.write({"kind": "heartbeat"})
        self.assertEqual(json.loads(output.path.read_text()), {"kind": "heartbeat"})
        self.assertEqual(output.path.stat().st_mode & 0o777, 0o644)
        with mock.patch.object(helper, "MAX_BYTES", 1):
            output.write({"kind": "heartbeat", "n": 2})
        self.assertTrue((self.folder / "events.jsonl.1").exists())
        self.assertEqual(len(output.path.read_text().splitlines()), 1)

    def test_resumes_from_its_cursor_or_starts_from_now(self):
        self.folder.mkdir()
        self.assertIn("--since=now", helper.command(self.folder))
        (self.folder / "journal.cursor").write_text("s=1")
        self.assertNotIn("--since=now", helper.command(self.folder))

    def test_a_heartbeat_at_once(self):
        output = helper.Output(self.folder)
        stop = threading.Event()
        stop.set()
        with mock.patch.object(stop, "is_set", side_effect=[False, True]):
            helper.beat(output, stop, every=0)
        self.assertIn("heartbeat", output.path.read_text())

    def test_main_writes_what_journalctl_says_then_ends(self):
        lines = [journal("NVRM: Xid (PCI:0000:01:00): 79, x") + "\n", journal("other") + "\n"]
        process = mock.MagicMock()
        process.__enter__.return_value.stdout = iter(lines)
        with mock.patch.object(helper.subprocess, "Popen", return_value=process), \
                mock.patch.object(helper, "beat"):
            self.assertEqual(helper.main([str(self.folder)]), 1)
        written = (self.folder / "events.jsonl").read_text().splitlines()
        self.assertEqual(len(written), 1)

    def test_main_needs_its_folder(self):
        with mock.patch("sys.stderr"):
            self.assertEqual(helper.main([]), 2)
