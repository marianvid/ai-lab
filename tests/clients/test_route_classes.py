"""A client's token reaches the model API only; the rest is for the page (radio-lab, 2026-11-27)."""

import unittest
from unittest import mock

from ai_lab.api.routes import RouteParts
from ai_lab.api.server import build_router
from ai_lab.clients.route_classes import model_api

# Every route of a full router that a token may call. A new route is management
# unless it is added here on purpose (and to the list in the request's ai-lab.md).
MODEL_API = {
    ("POST", "/v1/chat/completions"), ("POST", "/v1/completions"), ("POST", "/v1/messages"),
    ("POST", "/v1/messages/count_tokens"), ("POST", "/v1/audio/transcriptions"),
    ("POST", "/v1/audio/speech-segments"), ("POST", "/v1/audio/diarizations"),
    ("POST", "/v1/images/ocr"), ("POST", "/v1/audio/music/generations"),
    ("POST", "/v1/audio/music/window-generations"), ("POST", "/v1/audio/speech/generations"),
    ("POST", "/v1/audio/alignments"), ("POST", "/tokenize"), ("POST", "/v1/embeddings"),
    ("GET", "/v1/models"), ("GET", "/v1/models/{model}"), ("GET", "/v1/models/{vendor}/{model}"),
    ("POST", "/v1/images/generations"), ("POST", "/v1/images/edits"),
    ("GET", "/v1/audio/music/generations"),
    ("GET", "/v1/audio/music/generations/{generation_id}/audio"),
    ("GET", "/v1/audio/music/generations/{generation_id}/window-audio"),
    ("GET", "/v1/audio/music/generations/{generation_id}/bundle"),
    ("POST", "/v1/audio/music/generations/{generation_id}/pull"),
    ("POST", "/v1/audio/music/generations/{generation_id}/put-back"),
    ("DELETE", "/v1/audio/music/generations/{generation_id}"),
    ("DELETE", "/v1/audio/music/window-generations/{request_id}"),
    ("POST", "/api/clients/heartbeat"), ("GET", "/api/mailbox"), ("POST", "/api/mailbox"),
    ("GET", "/api/gateway"),
}


class RouteClassesTests(unittest.TestCase):
    def test_a_full_router_opens_to_a_token_exactly_the_model_api(self):
        parts = RouteParts(gateway=mock.MagicMock(), access=mock.MagicMock(),
                           journal=mock.MagicMock(), lanes=mock.MagicMock(),
                           images=mock.MagicMock(), gpu_health=mock.MagicMock())
        routes = set(build_router(mock.MagicMock(), parts).patterns())
        opened = {route for route in routes if model_api(*route)}
        self.assertEqual(opened, MODEL_API & routes)
        self.assertGreater(len(routes - opened), 60, "management stays the larger part")

    def test_management_is_refused_to_a_token_whatever_the_method(self):
        for method, path in (("GET", "/api/security"), ("PATCH", "/api/security"),
                             ("POST", "/api/security/password"), ("GET", "/api/clients"),
                             ("PATCH", "/api/gateway"), ("PATCH", "/api/setting-fields"),
                             ("POST", "/api/instances/qwen/load"), ("GET", "/api/instances"),
                             ("GET", "/api/journal"), ("GET", "/api/events"), ("POST", "/api/power")):
            with self.subTest(f"{method} {path}"):
                self.assertFalse(model_api(method, path))
        self.assertTrue(model_api("post", "/tokenize"))


if __name__ == "__main__":
    unittest.main()
