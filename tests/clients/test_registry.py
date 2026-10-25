"""The registered clients: a token shown once, kept only as a digest."""

import json
import stat
import tempfile
import unittest
from pathlib import Path

from ai_lab.clients.registry import ClientRegistry, digest


class RegistryTest(unittest.TestCase):
    def setUp(self):
        self.folder = Path(tempfile.mkdtemp())
        self.path = self.folder / "clients.json"

    def test_a_token_identifies_its_client_and_is_never_stored(self):
        registry = ClientRegistry(self.path)
        token = registry.register("scheduler-linux")
        self.assertEqual(registry.identify(token), "scheduler-linux")
        text = self.path.read_text()
        self.assertNotIn(token, text)
        self.assertIn(digest(token), text)

    def test_the_file_is_private(self):
        ClientRegistry(self.path).register("voice-studio")
        self.assertEqual(stat.S_IMODE(self.path.stat().st_mode), 0o600)

    def test_registering_again_replaces_the_token(self):
        registry = ClientRegistry(self.path)
        first = registry.register("marian-tests")
        second = registry.register("marian-tests")
        self.assertIsNone(registry.identify(first))
        self.assertEqual(registry.identify(second), "marian-tests")

    def test_clients_survive_a_restart(self):
        token = ClientRegistry(self.path).register("scheduler-mac")
        self.assertEqual(ClientRegistry(self.path).identify(token), "scheduler-mac")

    def test_removing_a_client_stops_its_token(self):
        registry = ClientRegistry(self.path)
        token = registry.register("voice-studio")
        registry.remove("voice-studio")
        self.assertIsNone(registry.identify(token))
        with self.assertRaises(KeyError):
            registry.remove("voice-studio")

    def test_names_never_show_tokens(self):
        registry = ClientRegistry(self.path)
        registry.register("b-client")
        registry.register("a-client")
        rows = registry.names()
        self.assertEqual([row["name"] for row in rows], ["a-client", "b-client"])
        self.assertNotIn("digest", json.dumps(rows))

    def test_a_bad_name_is_refused(self):
        with self.assertRaises(ValueError):
            ClientRegistry(self.path).register("Not Allowed")
        with self.assertRaises(ValueError):
            ClientRegistry(self.path).register("")

    def test_an_unknown_token_is_nobody(self):
        self.assertIsNone(ClientRegistry(self.path).identify("whatever"))


if __name__ == "__main__":
    unittest.main()
