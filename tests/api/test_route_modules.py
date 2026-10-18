"""Each small route module sends its paths to the right operation."""

import unittest

from ai_lab.api.router import Router
from ai_lab.api.routes import (RouteParts, catalog, downloads, huggingface, images, media,
                               register_all, settings, storage)
from ai_lab.api.routes.jobs import register_jobs
from ai_lab.types import Task


class Recorder:
    """Operations that answer every call with its name and arguments."""

    def __getattr__(self, name):
        def call(*args, **kwargs):
            return {"called": name, "args": list(args), "kwargs": kwargs}
        return call


class Jobs(Recorder):
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


class JobRouteTests(unittest.TestCase):
    def setUp(self):
        self.router = Router()
        images.register(self.router, Jobs())
        media.register(self.router, Jobs())

    def test_both_queues_list_read_and_cancel(self):
        for path in ("/api/image-jobs", "/api/media-jobs"):
            self.assertEqual(call(self.router, "GET", path)["called"], "list")
            self.assertEqual(call(self.router, "GET", path + "/j1")["args"], ["j1"])
            self.assertEqual(call(self.router, "DELETE", path + "/j1")["called"], "cancel")

    def test_any_queue_gets_the_three_routes(self):
        router = Router()
        register_jobs(router, "/api/other-jobs", Jobs())
        self.assertEqual(call(router, "DELETE", "/api/other-jobs/j2")["args"], ["j2"])

    def test_the_queues_routes_are_there_only_when_their_queue_is(self):
        bare = Router()
        register_all(bare, Recorder())
        self.assertIsNone(bare.match("GET", "/api/media-jobs"))
        both = Router()
        register_all(both, Recorder(), RouteParts(image_jobs=Jobs(), media_jobs=Jobs()))
        self.assertEqual(call(both, "GET", "/api/media-jobs")["called"], "list")
        self.assertEqual(call(both, "GET", "/api/image-jobs/j1")["args"], ["j1"])

    def test_image_requests_carry_their_task(self):
        made = call(self.router, "POST", "/v1/images/edits", body={"prompt": "x"})
        self.assertEqual(made["args"], [{"prompt": "x"}, Task.IMAGE_EDIT])
