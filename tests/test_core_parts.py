"""Small core modules, each on its own: configuration upgrades and policies,
the gateway's control and report, the progress stream and the upload shim."""

import unittest

from ai_lab import uploads
from ai_lab.api import uploads as api_uploads
from ai_lab.api.sse import stream
from ai_lab.config_migrations import SCHEMA_VERSION, migrate
from ai_lab.config_policy import GatewayPolicy, MediaPolicy
from ai_lab.events import EventBus
from ai_lab.gateway_control import GatewayControl
from ai_lab.gateway_errors import CardBusyError
from ai_lab.gateway_state import Shape
from ai_lab.gateway_stats import GatewayCounters, _runs, _waiting_summary
from ai_lab.types import ChangeEvent


class MigrationTests(unittest.TestCase):
    def test_an_unversioned_file_is_marked_current(self):
        self.assertEqual(migrate({"port": 1})["schema_version"], SCHEMA_VERSION)

    def test_a_newer_file_is_refused(self):
        with self.assertRaisesRegex(ValueError, "newer than this AI-Lab"):
            migrate({"schema_version": SCHEMA_VERSION + 1})

    def test_only_an_object_is_a_configuration(self):
        with self.assertRaises(ValueError):
            migrate([])  # type: ignore[arg-type]


class PolicyTests(unittest.TestCase):
    def test_gateway_defaults_and_task_overrides(self):
        policy = GatewayPolicy.from_mapping({"task_timeouts": {"ocr": {"first_byte_s": 60}}})
        self.assertEqual(policy.max_waiting, 150)
        self.assertEqual(policy.timeout_mapping(),
                         {"ocr": {"first_byte_s": 60.0, "between_bytes_s": 30.0}})

    def test_out_of_range_integers_name_their_section(self):
        with self.assertRaisesRegex(ValueError, "Gateway max_waiting must be between 1 and 10000"):
            GatewayPolicy.from_mapping({"max_waiting": 0})
        with self.assertRaisesRegex(ValueError, "Media max_queue"):
            MediaPolicy.from_mapping({"max_queue": 0})

    def test_unknown_media_settings_are_refused(self):
        with self.assertRaisesRegex(ValueError, "Unknown media settings: nope"):
            MediaPolicy.from_mapping({"nope": 1})


class Scheduler:
    def __init__(self, in_flight=0, waiting=(), switching=False):
        self.snapshot = {"in_flight": in_flight, "waiting": list(waiting),
                         "switching": switching,
                         "loaded": [{"shape": Shape.of("coder", {}), "in_flight": in_flight,
                                     "places": 4}]}

    def state(self):
        return self.snapshot


class ControlTests(unittest.TestCase):
    def test_an_idle_card_lets_the_buttons_through(self):
        control = GatewayControl(Scheduler())
        self.assertIsNone(control.busy())
        control.guard("unload", "coder")

    def test_a_busy_card_names_what_would_be_cut_off(self):
        control = GatewayControl(Scheduler(in_flight=2, waiting=[{}]))
        with self.assertRaisesRegex(CardBusyError, "coder is answering 2 requests and 1 more"):
            control.guard("unload", "coder")


class ReportTests(unittest.TestCase):
    def test_the_queue_reads_as_runs_and_per_model_counts(self):
        waiting = [{"shape": Shape.of("a", {}), "waiting_s": 3.0},
                   {"shape": Shape.of("a", {}), "waiting_s": 1.0},
                   {"shape": Shape.of("b", {}), "waiting_s": 2.0},
                   {"shape": Shape.of("a", {"ctx": 1}), "waiting_s": 0.5}]
        self.assertEqual([(run["instance_id"], run["requests"]) for run in _runs(waiting)],
                         [("a", 2), ("b", 1), ("a", 1)])
        self.assertEqual([(row["instance_id"], row["waiting"]) for row in _waiting_summary(waiting)],
                         [("a", 3), ("b", 1)])

    def test_counters_start_empty(self):
        counters = GatewayCounters()
        self.assertEqual((counters.requests, counters.last_error, counters.history), (0, "", []))


class Bus(EventBus):
    """A bus whose next subscriber already has events waiting."""

    def __init__(self, *waiting):
        super().__init__()
        self.waiting = waiting

    def subscribe(self):
        subscription = super().subscribe()
        for event in self.waiting:
            subscription._queue.put(event)
        return subscription


class StreamTests(unittest.TestCase):
    def test_events_are_written_until_the_client_goes(self):
        bus = Bus(ChangeEvent(topic="models"), ChangeEvent(topic="settings"))
        written: list[bytes] = []

        def write(data):
            if written:
                raise BrokenPipeError
            written.append(data)
        stream(bus, write, lambda: True)
        self.assertTrue(written[0].startswith(b"data: "))
        self.assertIn(b'"kind": "change"', written[0])
        self.assertEqual(bus.subscriber_count, 0)

    def test_nothing_is_written_once_the_client_has_gone(self):
        bus = Bus(ChangeEvent(topic="models"))
        written: list[bytes] = []
        stream(bus, written.append, lambda: False)
        self.assertEqual(written, [])


class UploadShimTests(unittest.TestCase):
    def test_the_old_import_path_is_the_same_check(self):
        self.assertIs(api_uploads.validate_image, uploads.validate_image)
        self.assertIs(api_uploads.UploadRejectedError, uploads.UploadRejectedError)
