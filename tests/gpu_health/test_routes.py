"""The page's routes call the right thing with the right arguments."""

import unittest

from ai_lab.api.router import Router
from ai_lab.api.routes import RouteParts, register_all
from ai_lab.api.routes import gpu_health as gpu_health_routes


class Health:
    def __init__(self):
        self.calls: list[tuple] = []

    def __getattr__(self, name):
        def called(*args):
            self.calls.append((name, *args))
            return {"called": name}
        return called


class Operations:
    def __getattr__(self, name):
        return lambda *args, **kwargs: None


def call(router, method, path, query=None, body=None):
    handler, captured = router.match(method, path)
    return handler(**captured, query=query, body=body)


class Routes(unittest.TestCase):
    def setUp(self):
        self.health = Health()
        self.router = Router()
        register_all(self.router, Operations(), RouteParts(gpu_health=self.health))

    def test_each_route(self):
        call(self.router, "GET", "/api/gpu-health")
        call(self.router, "GET", "/api/gpu-health/history", {"hours": "48"})
        call(self.router, "GET", "/api/gpu-health/report")
        call(self.router, "POST", "/api/gpu-health/baseline/reset", body={"note": "paste"})
        call(self.router, "POST", "/api/gpu-health/test-alarm")
        self.assertEqual(self.health.calls, [("status",), ("history", 48.0), ("report", 7),
                                             ("reset_baseline", "paste"), ("test_alarm",)])

    def test_a_bad_number_is_a_bad_request(self):
        with self.assertRaises(ValueError):
            call(self.router, "GET", "/api/gpu-health/history", {"hours": "many"})

    def test_the_module_registers_on_its_own(self):
        router = Router()
        gpu_health_routes.register(router, self.health)
        call(router, "GET", "/api/gpu-health")
        self.assertEqual(self.health.calls, [("status",)])

    def test_without_the_part_there_are_no_routes(self):
        router = Router()
        register_all(router, Operations())
        self.assertIsNone(router.match("GET", "/api/gpu-health"))
