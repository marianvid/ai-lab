import unittest

from ai_lab.events import EventBus, to_json
from ai_lab.types import Phase, RuntimeEvent


def event(elapsed=0):
    return RuntimeEvent(instance_id="qwen", phase=Phase.WEIGHTS_LOADING,
                        elapsed_ms=elapsed, progress=0.5,
                        memory_used_mb=100.0, memory_total_mb=32000.0)


class EventBusTests(unittest.TestCase):
    def test_every_subscriber_gets_every_event(self):
        bus = EventBus()
        first, second = bus.subscribe(), bus.subscribe()
        bus.publish(event())
        self.assertEqual(next(first.events(timeout=0.01)).instance_id, "qwen")
        self.assertEqual(next(second.events(timeout=0.01)).instance_id, "qwen")

    def test_a_slow_subscriber_never_blocks_the_publisher(self):
        """A browser that stops reading must not be able to stall a load."""
        bus = EventBus(queue_size=2)
        subscription = bus.subscribe()
        for index in range(50):
            bus.publish(event(index))
        stream = subscription.events(timeout=0.01)
        kept = [next(stream).elapsed_ms for _ in range(2)]
        self.assertEqual(kept, [48, 49])          # newest survive, oldest dropped

    def test_unsubscribing_stops_delivery(self):
        bus = EventBus()
        subscription = bus.subscribe()
        subscription.close()
        self.assertEqual(bus.subscriber_count, 0)
        bus.publish(event())
        self.assertIsNone(next(subscription.events(timeout=0.01)))

    def test_idle_yields_none_so_a_keepalive_can_be_sent(self):
        bus = EventBus()
        self.assertIsNone(next(bus.subscribe().events(timeout=0.01)))

    def test_serialisation_turns_the_phase_into_a_string(self):
        self.assertEqual(to_json(event())["phase"], "weights_loading")


if __name__ == "__main__":
    unittest.main()
