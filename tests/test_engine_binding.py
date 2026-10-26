"""Every engine's API listens on this machine only (list v6, item E).

Other machines reach the models only through AI-Lab's own port, which asks
for a token. The pages a person opens from a button are named separately.
"""

import unittest
from pathlib import Path

from ai_lab import network

ROOT = Path(network.__file__).resolve().parent
# The code that starts an engine's API, or is that API.
ENGINE_CODE = [*sorted((ROOT / "engines").rglob("*.py")),
               ROOT / "engine_host" / "http_host.py",
               ROOT / "speech" / "higgs_server.py"]


class EngineBindingTest(unittest.TestCase):
    def test_the_addresses(self):
        self.assertEqual(network.LOOPBACK, "127.0.0.1")
        self.assertEqual(network.ALL_INTERFACES, "0.0.0.0")
        self.assertEqual(network.local_url(8085, "/health"), "http://127.0.0.1:8085/health")

    def test_no_engine_api_listens_on_every_network_card(self):
        for path in ENGINE_CODE:
            text = path.read_text()
            with self.subTest(path=path.name):
                self.assertNotIn("ALL_INTERFACES", text)
                self.assertNotIn("PAGE_INTERFACES", text)
                self.assertNotIn("0.0.0.0", text)


if __name__ == "__main__":
    unittest.main()
