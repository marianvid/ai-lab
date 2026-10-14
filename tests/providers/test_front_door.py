"""Subscription and processor names reach their own lanes, everything else the card."""

import unittest
from http import HTTPStatus
from typing import Any, cast

from ai_lab.api.multipart import MultipartBody
from ai_lab.api.routes.gateway import _catalogue, _describe, _forwarder, _stats
from ai_lab.api.server import status_for
from ai_lab.lanes import SideLanes
from ai_lab.main import parse_arguments
from ai_lab.providers import ProviderRateLimitError, ProviderUnavailableError


class FakePool:
    def __init__(self):
        self.asked = []

    def serves(self, name):
        return name.startswith("claude/")

    def complete(self, path, payload):
        self.asked.append((path, payload["model"]))
        return {"object": "chat.completion"}

    def catalogue(self):
        return [{"id": "claude/sonnet", "object": "model"}]

    def stats(self):
        return {"anthropic": {"in_flight": 0}}


class FakeCpu:
    def __init__(self):
        self.asked = []
        self.stopped = False

    def serves(self, name):
        return name == "bge-m3"

    def embed(self, path, payload):
        self.asked.append((path, payload["model"]))
        return {"object": "list", "data": []}

    def catalogue(self):
        return [{"id": "bge-m3", "object": "model"}]

    def stats(self):
        return {"bge-m3": {"running": False}}

    def stop_all(self):
        self.stopped = True


class FakeGateway:
    def __init__(self):
        self.acquired = []

    def acquire(self, wanted, **_):
        self.acquired.append(wanted)
        raise LookupError("the card was asked")

    def catalogue(self):
        return []

    def stats(self):
        return {"in_flight": 0}

    def describe(self, name):
        raise KeyError(name)


class FrontDoorTests(unittest.TestCase):
    def setUp(self):
        self.pool: Any = FakePool()
        self.cpu: Any = FakeCpu()
        self.lanes = SideLanes(providers=cast(Any, self.pool), cpu=cast(Any, self.cpu))
        self.gateway: Any = FakeGateway()

    def test_a_subscription_name_never_touches_the_card(self):
        handle = _forwarder(self.gateway, "/v1/chat/completions", self.lanes)
        answer = handle(body={"model": "claude/sonnet", "messages": []})
        self.assertEqual(answer, {"object": "chat.completion"})
        self.assertEqual(self.pool.asked, [("/v1/chat/completions", "claude/sonnet")])
        self.assertEqual(self.gateway.acquired, [])

    def test_a_processor_name_never_touches_the_card(self):
        handle = _forwarder(self.gateway, "/v1/embeddings", self.lanes)
        answer = handle(body={"model": "bge-m3", "input": ["x"]})
        self.assertEqual(answer, {"object": "list", "data": []})
        self.assertEqual(self.cpu.asked, [("/v1/embeddings", "bge-m3")])
        self.assertEqual(self.gateway.acquired, [])

    def test_a_side_lane_takes_json_only(self):
        handle = _forwarder(self.gateway, "/v1/embeddings", self.lanes)
        upload = MultipartBody("multipart/form-data; boundary=b",
                               b'--b\r\nContent-Disposition: form-data; name="model"\r\n\r\n'
                               b"bge-m3\r\n--b--\r\n")
        with self.assertRaisesRegex(ValueError, "processor models take a JSON request"):
            handle(body=upload)

    def test_a_local_name_still_goes_to_the_card(self):
        handle = _forwarder(self.gateway, "/v1/chat/completions", self.lanes)
        with self.assertRaises(LookupError):
            handle(body={"model": "qwen36-nvfp4", "messages": []})
        self.assertEqual(self.gateway.acquired, ["qwen36-nvfp4"])

    def test_catalogue_description_and_stats_include_both_lanes(self):
        ids = [row["id"] for row in _catalogue(self.gateway, self.lanes)["data"]]
        self.assertEqual(ids, ["claude/sonnet", "bge-m3"])
        self.assertEqual(_describe(self.gateway, self.lanes, "bge-m3")["id"], "bge-m3")
        stats = _stats(self.gateway, self.lanes)
        self.assertEqual(stats["providers"], {"anthropic": {"in_flight": 0}})
        self.assertEqual(stats["cpu_models"], {"bge-m3": {"running": False}})
        self.assertNotIn("providers", _stats(self.gateway, SideLanes()))
        with self.assertRaises(KeyError):
            _describe(self.gateway, self.lanes, "nothing")

    def test_stopping_the_lanes_stops_the_processor_models(self):
        self.lanes.stop_all()
        self.assertTrue(self.cpu.stopped)
        SideLanes().stop_all()

    def test_provider_failures_map_to_http_statuses(self):
        self.assertEqual(status_for(ProviderRateLimitError("x")), HTTPStatus.TOO_MANY_REQUESTS)
        self.assertEqual(status_for(ProviderUnavailableError("x")), HTTPStatus.BAD_GATEWAY)


class ArgumentTests(unittest.TestCase):
    def test_the_configuration_is_required(self):
        with self.assertRaises(SystemExit):
            parse_arguments([])
        arguments = parse_arguments(["--config", "/etc/x.json", "--port", "9"])
        self.assertEqual((str(arguments.config), arguments.port), ("/etc/x.json", 9))


if __name__ == "__main__":
    unittest.main()
