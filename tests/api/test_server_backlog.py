import socket
import unittest
from http.server import BaseHTTPRequestHandler

from ai_lab.api.server import GatewayServer
from ai_lab.network import LOOPBACK


class BacklogTests(unittest.TestCase):
    def test_a_burst_of_connections_waits_instead_of_being_refused(self):
        """64 connections opened before the server accepts any of them.

        With the standard library's backlog of 5, most of these never
        complete; with the gateway's backlog they all wait their turn.
        """
        server = GatewayServer((LOOPBACK, 0), BaseHTTPRequestHandler)
        self.addCleanup(server.server_close)
        port = server.server_address[1]
        clients = []
        try:
            for _ in range(64):
                client = socket.create_connection((LOOPBACK, port), timeout=2)
                clients.append(client)
        finally:
            for client in clients:
                client.close()
        self.assertEqual(len(clients), 64)


if __name__ == "__main__":
    unittest.main()
