"""Each small route module sends its paths to the right operation."""

import unittest

from ai_lab.api.router import Router
from ai_lab.api.routes import (RouteParts, catalog, downloads, huggingface, images, runtime,
                               register_all, settings, storage)
from ai_lab.types import Task


class Recorder:
    """Operations that answer every call with its name and arguments."""

    def __getattr__(self, name):
        def call(*args, **kwargs):
            return {"called": name, "args": list(args), "kwargs": kwargs}
        return call


class Images(Recorder):
    pass


def call(router, method, path, **kwargs):
    matched = router.match(method, path)
    assert matched is not None, f"{method} {path} is not routed"
    handler, captured = matched
    return handler(**captured, **kwargs)


class RouteModuleTests(unittest.TestCase):
    def setUp(self):
        self.router = Router()
        self.operations = Recorder()
        for module in (catalog, downloads, huggingface, settings, storage):
            module.register(self.router, self.operations)

    def test_the_library(self):
        self.assertEqual(call(self.router, "GET", "/api/models", query={"engine": "vllm"})["args"],
                         ["vllm"])
        moved = call(self.router, "POST", "/api/models/gguf%2Fx/move",
                     body={"storage_tier": "fast"})
        self.assertEqual((moved["called"], moved["args"]), ("move_model", ["gguf/x", "fast"]))

    def test_downloads_and_hugging_face(self):
        started = call(self.router, "POST", "/api/downloads",
                       body={"repo": "a/b", "name": "b"})
        self.assertEqual(started["args"], ["a/b", "b", None, None])
        self.assertEqual(call(self.router, "GET", "/api/hf/search", query={"q": "qwen"})["args"],
                         ["qwen"])

    def test_settings_paths_installs_builds_and_folders(self):
        cases = {("PATCH", "/api/memory"): "update_memory",
                 ("POST", "/api/installs/vllm"): "install_engine",
                 ("POST", "/api/builds/llamacpp/update"): "update_engine",
                 ("GET", "/api/browse"): "browse",
                 ("GET", "/api/settings"): "settings_view"}
        for (method, path), operation in cases.items():
            self.assertEqual(call(self.router, method, path)["called"], operation, path)

    def test_storage(self):
        self.assertEqual(call(self.router, "DELETE", "/api/storage/cache")["args"], ["cache"])


class Card:
    """A gateway that records what the buttons asked of it."""

    def __init__(self):
        self.done = []

    def guard(self, action, instance_id):
        self.done.append(("guard", action, instance_id))

    def reset(self, reason):
        self.done.append(("reset", reason))

    def card_changed(self):
        self.done.append(("changed",))


class Started:
    def __init__(self, called):
        self.called = called

    def json(self):
        return {"called": self.called}


class Engines(Recorder):
    """Operations whose Load and Unload answer like a started operation."""

    def load(self, instance_id):
        return Started("load")

    def unload(self, instance_id):
        return Started("unload")


class RuntimeRouteTests(unittest.TestCase):
    def setUp(self):
        self.router = Router()
        self.card = Card()
        runtime.register(self.router, Engines(), self.card)

    def test_load_is_guarded_and_the_card_is_told(self):
        loaded = call(self.router, "POST", "/api/instances/qwen/load", body={})
        self.assertEqual(loaded["called"], "load")
        self.assertEqual(self.card.done, [("guard", "load", "qwen"), ("changed",)])

    def test_force_turns_the_queue_away_and_never_reaches_the_settings(self):
        applied = call(self.router, "POST", "/api/instances/qwen/apply",
                       body={"force": True, "context_size": 8192})
        self.assertEqual(applied["args"], ["qwen", {"context_size": 8192}])
        self.assertEqual(self.card.done[0], ("reset", "qwen was stopped by hand"))

    def test_without_a_gateway_nothing_is_guarded(self):
        router = Router()
        runtime.register(router, Engines())
        self.assertEqual(call(router, "POST", "/api/instances/qwen/unload")["called"], "unload")


class ImageRouteTests(unittest.TestCase):
    def setUp(self):
        self.router = Router()
        images.register(self.router, Images())

    def test_profiles_and_the_two_image_requests(self):
        self.assertEqual(call(self.router, "GET", "/api/image-profiles")["called"], "profiles")
        made = call(self.router, "POST", "/v1/images/edits", body={"prompt": "x"})
        self.assertEqual(made["args"], [{"prompt": "x"}, Task.IMAGE_EDIT])
        made = call(self.router, "POST", "/v1/images/generations", body={"prompt": "y"})
        self.assertEqual(made["args"], [{"prompt": "y"}, Task.IMAGE_GENERATION])

    def test_no_job_routes_remain(self):
        router = Router()
        register_all(router, Recorder(), RouteParts(images=Images()))
        for path in ("/api/image-jobs", "/api/media-jobs", "/api/reservations"):
            self.assertIsNone(router.match("GET", path), path)
        self.assertIsNotNone(router.match("GET", "/api/image-profiles"))
