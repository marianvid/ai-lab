"""The On/Off switch: what it refuses, what it runs, and that it survives a restart."""

import tempfile
import unittest
from pathlib import Path

from ai_lab.clients.power import Power, uses_a_model


class PowerTest(unittest.TestCase):
    def setUp(self):
        self.path = Path(tempfile.mkdtemp()) / "power.json"

    def test_on_by_default_and_off_survives_a_restart(self):
        power = Power(self.path, clock=lambda: 5.0)
        self.assertEqual(power.view(), {"on": True, "since": None})
        power.switch(False)
        self.assertEqual(Power(self.path).view(), {"on": False, "since": 5.0})

    def test_switching_off_runs_the_off_actions_once(self):
        power = Power(self.path)
        ran = []
        power.when_off(lambda: ran.append("reset"))
        power.switch(False)
        power.switch(False)
        power.switch(True)
        self.assertEqual(ran, ["reset"])

    def test_what_is_refused_while_off(self):
        self.assertTrue(uses_a_model("POST", "/v1/chat/completions"))
        self.assertTrue(uses_a_model("POST", "/tokenize"))
        self.assertFalse(uses_a_model("POST", "/api/media-jobs"))
        self.assertTrue(uses_a_model("POST", "/api/instances/qwen/load"))
        self.assertFalse(uses_a_model("GET", "/v1/models"))
        self.assertFalse(uses_a_model("POST", "/api/instances/qwen/unload"))
        self.assertFalse(uses_a_model("POST", "/api/power"))


if __name__ == "__main__":
    unittest.main()
