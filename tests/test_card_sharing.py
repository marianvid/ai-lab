"""Embeddings borrow the card; every card load asks them to step aside first."""

import unittest
from types import SimpleNamespace
from typing import Any

from ai_lab.gateway.card_sharing import NoNeighbours, guard_loads
from ai_lab.config.store import Config, Instance, ModelRoot
from ai_lab.config.validation import validate_configuration
from ai_lab.wiring import CONFIG_SECTIONS
from ai_lab.gateway.resources import GatewayResources
from ai_lab.wiring import card_free_mb as _card_free_mb
from ai_lab.types import AcceleratorSnapshot
from ai_lab.operations import Operations
from ai_lab.operations_services import build_services
from tests import test_operations as base


class Neighbours:
    def __init__(self, held_mb=0.0):
        self.held_mb = held_mb
        self.calls = []

    def make_way(self):
        self.calls.append("make_way")
        self.held_mb = 0.0

    def settled(self):
        self.calls.append("settled")

    def reclaimable_mb(self):
        return self.held_mb


def snapshot(kind="dedicated", used=28000.0, total=32000.0):
    return AcceleratorSnapshot(available=True, name="card", kind="cuda", memory_kind=kind,
                               memory_used_mb=used, memory_total_mb=total)


class ResourcesTests(unittest.TestCase):
    def setUp(self):
        self.neighbours = Neighbours(held_mb=1300.0)
        self.resources = GatewayResources(SimpleNamespace(accelerator=lambda: snapshot(kind="unified")),
                                          500, 1, 0.01,
                                          neighbours=self.neighbours)

    def test_what_a_neighbour_holds_counts_as_free_for_a_card_model(self):
        pools = {"card": {"available_mb": 4000}}
        self.assertEqual(self.resources.free_mb(pools), 5300.0)

    def test_neighbours_leave_before_the_card_is_read_even_when_nothing_was_stopped(self):
        self.resources.wait_until_quiet([])
        self.assertEqual(self.neighbours.calls, ["make_way"])
        self.assertEqual(self.resources.free_mb({"card": {"available_mb": 4000}}), 4000.0)


class LoadTests(unittest.TestCase):
    operations: Any
    store: Any
    host: Any
    setUp = base.OperationsTests.setUp

    def test_a_load_asks_the_neighbours_to_step_aside_and_tells_them_when_done(self):
        neighbours = Neighbours()
        parts = base.parts_on(self.store, self.host)
        services = build_services(parts)
        guard_loads(services.instance_service, neighbours)
        self.assertTrue(Operations(parts, services).load("qwen").ok)
        self.assertEqual(neighbours.calls, ["make_way", "settled"])

    def test_a_load_that_fails_still_says_it_is_over(self):
        neighbours = Neighbours()

        def broken(*_args):
            raise RuntimeError("engine would not start")
        service = SimpleNamespace(load=broken)
        guard_loads(service, neighbours)
        with self.assertRaisesRegex(RuntimeError, "would not start"):
            service.load("qwen")
        self.assertEqual(neighbours.calls, ["make_way", "settled"])


class CardRoomTests(unittest.TestCase):
    def test_free_memory_only_on_a_card_of_its_own(self):
        self.assertEqual(_card_free_mb(SimpleNamespace(accelerator=snapshot)), 4000.0)
        unified = SimpleNamespace(accelerator=lambda: snapshot(kind="unified"))
        self.assertIsNone(_card_free_mb(unified))
        self.assertIsNone(_card_free_mb(SimpleNamespace(accelerator=lambda: snapshot(total=0))))

        def broken():
            raise OSError("no driver")
        self.assertIsNone(_card_free_mb(SimpleNamespace(accelerator=broken)))

    def test_nobody_borrows_the_card_by_default(self):
        nobody = NoNeighbours()
        nobody.make_way()
        nobody.settled()
        self.assertEqual(nobody.reclaimable_mb(), 0.0)

    def test_the_card_port_may_not_be_an_instances(self):
        config = Config(port=8090, models_root="/models",
                        model_roots=[ModelRoot("core", "Core", "/models")],
                        instances=[Instance("qwen", "vllm", "core/qwen", 8391)],
                        cpu_models={"binary": "/bin/llama-server", "models": {"bge-m3": {
                            "path": "/m/bge.gguf", "port": 8390,
                            "card": {"port": 8391, "needs_mb": 1300}}}})
        with self.assertRaisesRegex(ValueError, "port 8391 is already assigned"):
            validate_configuration(config, {"vllm"}, sections=CONFIG_SECTIONS)


if __name__ == "__main__":
    unittest.main()
