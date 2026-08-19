"""The web layer, exercised over a real socket.

Started on port 0 so the operating system picks a free port and the suite can
run anywhere, including alongside a real AI-Lab.
"""

import json
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

from ai_lab.api.server import Handler, build_router
from ai_lab.events import EventBus


class FakeOperations:
    def __init__(self):
        self.calls = []

    def settings_view(self):
        return {"title": "AI-Lab"}

    def models(self, engine_id=None):
        self.calls.append(("models", engine_id))
        return [{"id": "gguf/a", "name": "a"}]

    def instances(self):
        return [{"id": "qwen"}]

    def load(self, instance_id):
        self.calls.append(("load", instance_id))
        return _Operation()

    def unload(self, instance_id):
        raise KeyError(instance_id)

    def apply_and_reload(self, instance_id, changes):
        self.calls.append(("apply", instance_id, changes))
        return {"id": instance_id, "applied": True}

    def new_instance_form(self):
        return {"port": 8082, "engines": [], "models": []}

    def create_instance(self, payload):
        raise ValueError("Port 8080 is already in use")

    def update_instance(self, instance_id, changes):
        return {"id": instance_id}

    def delete_instance(self, instance_id):
        return None

    def transfers(self):
        return []

    def download(self, repo, name, repository_id):
        return {"id": "t1"}

    def cancel_download(self, transfer_id):
        return None

    def search(self, query, only_supported=True):
        self.calls.append(("search", query, only_supported))
        return []

    def remote_sets(self, repo, only_supported=True):
        return []


class _Operation:
    def json(self):
        return {"ok": True, "total_ms": 42}


class ServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.operations = FakeOperations()
        handler = type("TestHandler", (Handler,), {
            "router": build_router(cls.operations), "bus": EventBus()})
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        cls.base = f"http://127.0.0.1:{cls.server.server_address[1]}"
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def call(self, method, path, body=None):
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(self.base + path, data=data, method=method,
                                         headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read())

    def test_a_get_returns_json(self):
        status, payload = self.call("GET", "/api/instances")
        self.assertEqual(status, 200)
        self.assertEqual(payload, [{"id": "qwen"}])

    def test_a_query_string_reaches_the_handler(self):
        self.call("GET", "/api/models?engine=llamacpp")
        self.assertIn(("models", "llamacpp"), self.operations.calls)

    def test_a_path_parameter_reaches_the_handler(self):
        status, payload = self.call("POST", "/api/instances/qwen/load")
        self.assertEqual(status, 200)
        self.assertEqual(payload["total_ms"], 42)
        self.assertIn(("load", "qwen"), self.operations.calls)

    def test_a_body_reaches_the_handler(self):
        self.call("POST", "/api/instances/qwen/apply", {"params": {"context_size": 8192}})
        self.assertIn(("apply", "qwen", {"params": {"context_size": 8192}}),
                      self.operations.calls)

    def test_a_missing_thing_is_a_404(self):
        status, payload = self.call("POST", "/api/instances/nope/unload")
        self.assertEqual(status, 404)
        self.assertIn("nope", payload["error"])

    def test_a_rejected_change_explains_itself(self):
        """The message is shown to a person, so it must survive the round trip."""
        status, payload = self.call("POST", "/api/instances", {"id": "x"})
        self.assertEqual(status, 400)
        self.assertEqual(payload["error"], "Port 8080 is already in use")

    def test_an_unknown_route_is_a_404(self):
        status, _ = self.call("POST", "/api/nothing-here")
        self.assertEqual(status, 404)

    def test_the_interface_is_served(self):
        with urllib.request.urlopen(self.base + "/", timeout=5) as response:
            body = response.read().decode()
        self.assertEqual(response.status, 200)
        self.assertIn("<title>AI-Lab</title>", body)

    def test_static_files_cannot_escape_the_web_directory(self):
        status, _ = self.call("GET", "/../config.json")
        self.assertEqual(status, 404)

    def test_the_event_stream_announces_itself_correctly(self):
        request = urllib.request.Request(self.base + "/api/events")
        with urllib.request.urlopen(request, timeout=5) as response:
            self.assertEqual(response.headers["Content-Type"], "text/event-stream")


if __name__ == "__main__":
    unittest.main()


class CachingTests(unittest.TestCase):
    """Nothing may be cached by the browser.

    With no cache headers a browser caches static files by its own guesswork,
    so a deployment reaches the server while the page in front of you stays as
    it was — and refreshing does not help, which is a confusing way to lose an
    afternoon.
    """

    @classmethod
    def setUpClass(cls):
        handler = type("CacheHandler", (Handler,), {
            "router": build_router(FakeOperations()), "bus": EventBus()})
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        cls.base = f"http://127.0.0.1:{cls.server.server_address[1]}"
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def headers_for(self, path):
        with urllib.request.urlopen(self.base + path, timeout=5) as response:
            return response.headers

    def test_the_interface_is_never_cached(self):
        self.assertIn("no-store", self.headers_for("/").get("Cache-Control", ""))

    def test_scripts_are_never_cached(self):
        self.assertIn("no-store", self.headers_for("/js/app.js").get("Cache-Control", ""))

    def test_api_responses_are_never_cached(self):
        self.assertIn("no-store", self.headers_for("/api/instances").get("Cache-Control", ""))
