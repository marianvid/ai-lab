"""The web layer, exercised over a real socket.

Started on port 0 so the operating system picks a free port and the suite can
run anywhere, including alongside a real AI-Lab.
"""

import json
import threading
import unittest
import urllib.error
import urllib.request
from http import HTTPStatus
from http.server import ThreadingHTTPServer

from ai_lab.api.server import Handler, build_router
from ai_lab.api.server import status_for
from ai_lab.gateway import CardBusy, NotConfigured
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

    def test_a_message_written_as_a_sentence_arrives_as_one(self):
        # KeyError renders its argument with repr(), so "Unknown instance:
        # nope" would reach the screen wrapped in quotes. Six places raise one
        # that way and this is where they are all unwrapped.
        status, payload = self.call("POST", "/api/instances/nope/unload")
        self.assertEqual(status, 404)
        self.assertEqual(payload["error"], "nope",
                         f"arrived as {payload['error']!r}")

    def test_a_subclass_lands_on_its_parent_rule(self):
        # Looked up by exact type, a KeyError subclass misses the 404 rule and
        # leaves as a bad request. The gateway raises one for a model name
        # nobody serves, and 400 tells an agent the wrong thing about its own
        # request: it would retry differently instead of fixing the name.
        self.assertEqual(status_for(NotConfigured("no such model")),
                         HTTPStatus.NOT_FOUND)

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


class BusyCardTests(unittest.TestCase):
    """The whole round trip of the refusal, because the page acts on it.

    A message alone is not enough. The page has to be able to tell this
    refusal from every other one, and to name the model that is working — so
    the status code and the detail have to survive the trip out.
    """

    class Gateway:
        """Busy until told otherwise, which is all the routes need of it."""

        def __init__(self):
            self.holder = {"instance_id": "coder", "answering": True}
            self.guarded = []

        def guard(self, action, instance_id):
            self.guarded.append((action, instance_id))
            if self.holder is None:
                return
            raise CardBusy(f"coder is answering a request; {action} on "
                           f"{instance_id} would cut it off", self.holder)

    @classmethod
    def setUpClass(cls):
        cls.operations = FakeOperations()
        cls.gateway = cls.Gateway()
        handler = type("BusyHandler", (Handler,), {
            "router": build_router(cls.operations, cls.gateway),
            "bus": EventBus()})
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        cls.base = f"http://127.0.0.1:{cls.server.server_address[1]}"
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self):
        self.gateway.holder = {"instance_id": "coder", "answering": True}
        self.operations.calls.clear()
        self.gateway.guarded.clear()

    call = ServerTests.call

    def test_loading_while_the_card_is_busy_is_refused(self):
        status, payload = self.call("POST", "/api/instances/qwen/load")
        self.assertEqual(status, 409, "409 rather than 400: the same request "
                                      "will work once the card is free")
        self.assertEqual(payload["busy"]["instance_id"], "coder")
        self.assertNotIn(("load", "qwen"), self.operations.calls)

    def test_the_refusal_says_which_model_is_working(self):
        _, payload = self.call("POST", "/api/instances/qwen/load")
        self.assertTrue(payload["busy"]["answering"])
        self.assertIn("coder", payload["error"])

    def test_force_goes_ahead_anyway(self):
        status, _ = self.call("POST", "/api/instances/qwen/load", {"force": True})
        self.assertEqual(status, 200)
        self.assertIn(("load", "qwen"), self.operations.calls)

    def test_force_is_not_saved_as_a_setting(self):
        # `apply` writes the body into the instance's configuration. The
        # override is an instruction to the web layer and has no business
        # being stored beside the engine's own settings.
        self.call("POST", "/api/instances/qwen/apply",
                  {"force": True, "params": {"context_size": 8192}})
        self.assertIn(("apply", "qwen", {"params": {"context_size": 8192}}),
                      self.operations.calls)

    def test_a_free_card_is_not_asked_about_twice(self):
        self.gateway.holder = None
        status, _ = self.call("POST", "/api/instances/qwen/load")
        self.assertEqual(status, 200)
        self.assertEqual(self.gateway.guarded, [("load", "qwen")])

    def test_reading_the_list_is_never_refused(self):
        status, _ = self.call("GET", "/api/instances")
        self.assertEqual(status, 200)
        self.assertEqual(self.gateway.guarded, [],
                         "looking at the page must not be guarded")


class SettingsFieldTests(unittest.TestCase):
    """The field carrying startup settings must not reach the engine.

    The engine does not know it, and would ignore it without a word — which is
    the fault this whole field exists to avoid.
    """

    class Gateway:
        def __init__(self):
            self.asked = []

        def acquire(self, wanted, shape=None, settings=None):
            self.asked.append((wanted, shape, settings))
            return SettingsFieldTests.Lease(self)

        def release(self):
            pass

        def stats(self):
            return {}

        def catalogue(self):
            return []

    class Lease:
        def __init__(self, gateway):
            self.gateway = gateway
            self.instance_id = "qwen"
            self.port = 9
            self.model_name = "qwen-real"

    def test_the_field_is_read_and_removed(self):
        from ai_lab.api.routes.gateway import SETTINGS_FIELD, _forwarder
        gateway = self.Gateway()
        forwarded = {}

        def fake_forward(url, payload, on_close=None):
            forwarded.update(payload)
            if on_close:
                on_close()
            return None

        import ai_lab.api.routes.gateway as module
        original, module.forward = module.forward, fake_forward
        try:
            handle = _forwarder(gateway, "/v1/chat/completions")
            handle(body={"model": "qwen", "messages": [], "temperature": 0.5,
                         SETTINGS_FIELD: {"context_size": 65536}})
        finally:
            module.forward = original

        self.assertEqual(gateway.asked[0][2], {"context_size": 65536},
                         "the settings never reached the gateway")
        self.assertNotIn(SETTINGS_FIELD, forwarded,
                         "the field was forwarded to the engine")
        self.assertEqual(forwarded["temperature"], 0.5,
                         "the rest of the body must go through untouched")
        self.assertEqual(forwarded["model"], "qwen-real")

    def test_a_field_that_is_not_an_object_is_refused(self):
        from ai_lab.api.routes.gateway import SETTINGS_FIELD, _forwarder
        handle = _forwarder(self.Gateway(), "/v1/chat/completions")
        with self.assertRaises(ValueError) as caught:
            handle(body={"model": "qwen", "messages": [],
                         SETTINGS_FIELD: "context_size=65536"})
        self.assertIn(SETTINGS_FIELD, str(caught.exception))


class NoGatewayTests(unittest.TestCase):
    """A router built without a gateway still starts and stops models.

    The tests build one that way, and so does anything embedding this without
    the agent front door. Nothing is being served, so there is nothing to
    interrupt and nothing to guard.
    """

    def test_loading_works_with_no_gateway_wired(self):
        operations = FakeOperations()
        router = build_router(operations)
        handler, captured = router.match("POST", "/api/instances/qwen/load")
        self.assertEqual(handler(query={}, body={}, **captured)["total_ms"], 42)
