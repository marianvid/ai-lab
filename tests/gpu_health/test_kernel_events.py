"""The host helper's file: new lines only, rotation, Xid and AER, the heartbeat."""

import json
import os
import tempfile
import unittest
from pathlib import Path

from ai_lab.gpu_health.kernel.events import KernelEvents, parse

from .fakes import settings

XID_79 = "NVRM: Xid (PCI:0000:01:00): 79, pid=1234, GPU has fallen off the bus."
XID_13 = "NVRM: Xid (PCI:0000:01:00): 13, Graphics Exception"
AER_FATAL = "nvidia 0000:01:00.0: AER: PCIe Bus Error: severity=Uncorrected (Fatal), type=Transaction Layer"
AER_CORRECTED = "pcieport 0000:00:01.1: AER: Corrected error received: 0000:01:00.0"
AER_OTHER = "nvme 0000:02:00.0: AER: PCIe Bus Error: severity=Uncorrected (Non-Fatal)"


class Parse(unittest.TestCase):
    def test_critical_and_ordinary_xids(self):
        found = parse(XID_79, settings())[0]
        self.assertEqual((found.key, found.level, found.detail["xid"]), ("xid:79", "critical", 79))
        self.assertIn("fallen off the bus", found.message)
        self.assertEqual(parse(XID_13, settings())[0].level, "warning")

    def test_aer_on_the_card_only(self):
        self.assertEqual(parse(AER_FATAL, settings())[0].level, "critical")
        self.assertEqual(parse(AER_CORRECTED, settings())[0].level, "info")
        self.assertEqual(parse(AER_OTHER, settings()), [])
        self.assertEqual(len(parse(AER_OTHER, settings(pci_address=""))), 1)
        self.assertEqual(len(parse(AER_FATAL, settings(pci_address="01:00.0"))), 1)

    def test_anything_else(self):
        self.assertEqual(parse("usb 1-1: new device", settings()), [])


class Follow(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.path = Path(folder.name) / "events.jsonl"
        self.events = KernelEvents(settings(helper_events_path=str(self.path)))

    def write(self, *entries, mode="a"):
        with self.path.open(mode) as handle:
            for entry in entries:
                handle.write(entry if isinstance(entry, str) else json.dumps(entry) + "\n")

    def test_only_new_complete_lines_are_read(self):
        self.assertEqual(self.events.read(), [])  # no file yet
        self.write({"ts": 1, "kind": "kernel", "line": XID_79}, '{"ts": 2, "kind": "ker')
        self.assertEqual([item.key for item in self.events.read()], ["xid:79"])
        self.write('nel", "line": "' + XID_13 + '"}\n')
        self.assertEqual([item.key for item in self.events.read()], ["xid:13"])
        self.assertEqual(self.events.read(), [])

    def test_a_rotated_file_is_read_from_the_start(self):
        self.write({"ts": 1, "kind": "kernel", "line": XID_13})
        self.events.read()
        os.replace(self.path, self.path.with_suffix(".1"))
        self.write({"ts": 2, "kind": "kernel", "line": XID_79}, mode="w")
        self.assertEqual([item.key for item in self.events.read()], ["xid:79"])

    def test_lines_written_just_before_the_rotation_are_not_lost(self):
        self.write({"ts": 1, "kind": "kernel", "line": XID_13})
        self.events.read()
        self.write({"ts": 2, "kind": "kernel", "line": XID_79})
        os.replace(self.path, self.path.with_name("events.jsonl.1"))
        self.write({"ts": 3, "kind": "kernel", "line": AER_FATAL}, mode="w")
        self.assertEqual([item.key for item in self.events.read()], ["xid:79", "aer_uncorrectable"])

    def test_bad_lines_are_skipped(self):
        self.write("not json\n", "[1, 2]\n", {"ts": 1, "kind": "kernel", "line": XID_13})
        self.assertEqual(len(self.events.read()), 1)

    def test_the_heartbeat(self):
        self.assertEqual(self.events.silent(100.0)[0].key, "helper_silent")
        self.write({"ts": 100.0, "kind": "heartbeat"})
        self.events.read()
        self.assertEqual(self.events.silent(100.0 + 7200), [])
        self.assertIn("silent for 2.0 h", self.events.silent(100.0 + 7201)[0].message)

    def test_the_position_survives_a_restart(self):
        self.write({"ts": 1, "kind": "kernel", "line": XID_13})
        self.events.read()
        again = KernelEvents(settings(helper_events_path=str(self.path)), self.events.position)
        self.assertEqual(again.read(), [])

    def test_without_a_helper_nothing_is_watched_or_missed(self):
        events = KernelEvents(settings())
        self.assertFalse(events.installed)
        self.assertEqual((events.read(), events.silent(1e9)), ([], []))
