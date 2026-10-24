"""The status command: it asks the running AI-Lab and prints plain lines."""

import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock

from ai_lab.gpu_health.output import cli

STATE = {
    "available": True, "running": True, "last_ok": 1_800_000_000.0,
    "current": {"temp_gpu": 28, "power_w": 9.6, "model": "qwen"},
    "not_available": ["temp_hotspot", "temp_mem"],
    "analysis": {"idle_today": {"temp": 28.5, "from_day": "2026-10-24"},
                 "baselines": {"global": {"r": 0.2, "windows": 12, "period_start": "a",
                                          "period_end": "b"}}},
    "alarms": [{"level": "alarm", "message": "Chip hot"}],
    "helper": {"installed": False, "last_heartbeat": None},
    "failures": {"slow": {"count": 2, "error": "NvmlError: x"}},
    "counts": {"samples": 5},
}


def run(*argv, answer=None, error=None):
    out, err = io.StringIO(), io.StringIO()
    with mock.patch.object(cli, "fetch", side_effect=error, return_value=answer) as fetch, \
            redirect_stdout(out), redirect_stderr(err):
        code = cli.main(list(argv))
    return code, out.getvalue(), err.getvalue(), fetch


class Command(unittest.TestCase):
    def test_status(self):
        code, out, _, fetch = run("status", answer=STATE)
        self.assertEqual(code, 0)
        fetch.assert_called_once_with(cli.DEFAULT_URL + "/api/gpu-health")
        for expected in ("running", "temp_gpu 28", "temp_hotspot, temp_mem", "28.5 °C",
                         "R 0.200", "ALARM: Chip hot", "not installed", "failed 2 times"):
            self.assertIn(expected, out)

    def test_status_without_a_reference_or_a_card(self):
        state = {**STATE, "analysis": {"idle_today": None, "baselines": {}},
                 "helper": {"installed": True, "last_heartbeat": None}}
        out = run(answer=state)[1]
        self.assertIn("not formed yet", out)
        self.assertIn("last heartbeat never", out)
        self.assertIn("not available here", run(answer={"available": False, "reason": "Mac"})[1])

    def test_report_and_json(self):
        code, out, _, fetch = run("report", "--days", "30", "--url", "http://h:1/",
                                  answer={"text": "GPU health, last 30 days"})
        fetch.assert_called_once_with("http://h:1/api/gpu-health/report?days=30")
        self.assertEqual(out.strip(), "GPU health, last 30 days")
        self.assertEqual(json.loads(run("--json", answer={"a": 1})[1]), {"a": 1})

    def test_summary_and_alerts(self):
        short = {"available": True, "health": "alarm", "monitor": "running", "last_reading": None,
                 "last_alert_id": 7, "alarms": [{"level": "alarm", "message": "Chip hot"}]}
        code, out, _, fetch = run("summary", answer=short)
        fetch.assert_called_once_with(cli.DEFAULT_URL + "/api/gpu-health/summary")
        self.assertIn("ALARM · monitor running", out)
        self.assertIn("last alert id 7", out)
        self.assertIn("not available here", run("summary", answer={"available": False})[1])
        listed = {"alerts": [{"id": 8, "ts": None, "level": "critical", "kind": "event",
                              "message": "Xid 79"}], "last_id": 8, "more": True}
        code, out, _, fetch = run("alerts", "--after", "7", answer=listed)
        fetch.assert_called_once_with(cli.DEFAULT_URL + "/api/gpu-health/alerts?after=7")
        self.assertIn("8 never CRITICAL [event] Xid 79", out)
        self.assertIn("last id 8 (more", out)

    def test_no_answer(self):
        code, _, err, _ = run(error=OSError("refused"))
        self.assertEqual(code, 1)
        self.assertIn("did not answer", err)

    def test_fetch_reads_json(self):
        answer = mock.MagicMock()
        answer.__enter__.return_value.read.return_value = b'{"ok": true}'
        with mock.patch("urllib.request.urlopen", return_value=answer):
            self.assertEqual(cli.fetch("http://x"), {"ok": True})
        self.assertEqual(cli.when(None), "never")
