"""Subscription names reach the provider pool, everything else the card."""

import unittest
from http import HTTPStatus
from typing import Any

from ai_lab.api.routes.gateway import _catalogue, _describe, _forwarder, _stats
from ai_lab.api.server import status_for
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
        self.gateway: Any = FakeGateway()

    def test_a_subscription_name_never_touches_the_card(self):
        handle = _forwarder(self.gateway, "/v1/chat/completions", self.pool)
        answer = handle(body={"model": "claude/sonnet", "messages": []})
        self.assertEqual(answer, {"object": "chat.completion"})
        self.assertEqual(self.pool.asked, [("/v1/chat/completions", "claude/sonnet")])
        self.assertEqual(self.gateway.acquired, [])

    def test_a_local_name_still_goes_to_the_card(self):
        handle = _forwarder(self.gateway, "/v1/chat/completions", self.pool)
        with self.assertRaises(LookupError):
            handle(body={"model": "qwen36-nvfp4", "messages": []})
        self.assertEqual(self.gateway.acquired, ["qwen36-nvfp4"])

    def test_catalogue_description_and_stats_include_the_providers(self):
        self.assertEqual(_catalogue(self.gateway, self.pool)["data"][0]["id"], "claude/sonnet")
        self.assertEqual(_describe(self.gateway, self.pool, "claude/sonnet")["id"],
                         "claude/sonnet")
        self.assertEqual(_stats(self.gateway, self.pool)["providers"],
                         {"anthropic": {"in_flight": 0}})
        self.assertNotIn("providers", _stats(self.gateway, None))

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
