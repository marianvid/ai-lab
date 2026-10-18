"""The mlx-lm launcher's two changes, on a stand-in for mlx-lm's server."""

import io
import sys
import types
import unittest
from typing import Any
from unittest import mock

from ai_lab.text import mlxlm_server


class Handler:
    def __init__(self):
        self.status = None
        self.wfile = io.BytesIO()

    def _set_completion_headers(self, status):
        self.status = status

    def end_headers(self):
        pass

    def handle_health_check(self):
        self.status = 200


class Provider:
    def __init__(self):
        self.loaded = []

    def load(self, model_path, adapter_path=None, draft_model_path=None):
        self.loaded.append(model_path)

    def load_default(self):
        self.loaded.append("default")


def patched_server() -> Any:
    server: Any = types.ModuleType("mlx_lm.server")
    server.ModelProvider = type("ModelProvider", (Provider,), {})
    server.APIHandler = type("APIHandler", (Handler,), {})
    server.main = lambda: None
    package: Any = types.ModuleType("mlx_lm")
    package.server = server
    with mock.patch.dict(sys.modules, {"mlx_lm": package, "mlx_lm.server": server}):
        mlxlm_server.main()
    return server


class MlxLmServerTests(unittest.TestCase):
    def test_health_says_loading_until_the_weights_are_in(self):
        server = patched_server()
        handler, provider = server.APIHandler(), server.ModelProvider()
        server.APIHandler.handle_health_check(handler)
        self.assertEqual(handler.status, 503)
        self.assertIn(b"loading", handler.wfile.getvalue())
        server.ModelProvider.load_default(provider)
        server.APIHandler.handle_health_check(handler)
        self.assertEqual(handler.status, 200)

    def test_any_model_name_means_the_one_this_process_holds(self):
        server = patched_server()
        provider = server.ModelProvider()
        server.ModelProvider.load(provider, "some/hub-name")
        self.assertEqual(provider.loaded, ["default_model"])
