"""The pieces the gateway and its scheduler are built from, each on its own."""

import dataclasses
import unittest

from ai_lab.gateway.catalogue import GatewayCatalogue
from ai_lab.gateway.errors import CouldNotLoadError, NotConfiguredError
from ai_lab.gateway.limits import FIRST_BYTE_S, MAX_WAITING, GatewayLimits
from ai_lab.gateway.loading import CardLoading
from ai_lab.gateway.state import Shape
from ai_lab.gateway.scheduler_parts import (
    AbandonedError,
    AdmissionWithdrawnError,
    Waiting,
    release,
)
from ai_lab.gateway.scheduler_planning import Planner, QueueState


class Entries:
    """Operations reduced to the configured entries."""

    def __init__(self, *ids):
        self.entries = [{"id": identifier, "model_id": f"gguf/x/{identifier}"}
                        for identifier in ids]

    def configured(self):
        return self.entries


class LimitsTests(unittest.TestCase):
    def test_the_defaults_are_the_machine_s_starting_values(self):
        limits = GatewayLimits()
        self.assertEqual((limits.first_byte_s, limits.max_waiting), (FIRST_BYTE_S, MAX_WAITING))
        self.assertEqual(limits.task_timeouts, {})

    def test_limits_do_not_change_once_made(self):
        with self.assertRaises(dataclasses.FrozenInstanceError):
            GatewayLimits().max_waiting = 3  # type: ignore[misc]


class CatalogueTests(unittest.TestCase):
    def setUp(self):
        self.catalogue = GatewayCatalogue()
        self.catalogue.operations = Entries("coder", "reviewer")  # type: ignore[assignment]

    def test_an_entry_answers_to_its_id_in_any_case(self):
        self.assertEqual(self.catalogue.resolve(" Coder ")["id"], "coder")

    def test_an_unknown_name_lists_what_is_known(self):
        with self.assertRaisesRegex(NotConfiguredError, "Known: coder, reviewer"):
            self.catalogue.resolve("nobody")

    def test_the_engine_s_name_is_the_last_segment_of_the_model(self):
        self.assertEqual(GatewayCatalogue._engine_name({"model_id": "gguf/qwen/Q3"}), "Q3")


class Resources:
    def __init__(self, needs_mb, free_mb):
        self.needs, self.free = needs_mb, free_mb

    def needs_mb(self, _shape, _card_total_mb):
        return self.needs

    def card_total_mb(self):
        return 32000.0

    def free_mb(self, _pools):
        return self.free


class LoadingTests(unittest.TestCase):
    def loading(self, needs_mb, free_mb):
        loading = CardLoading()
        loading.resources = Resources(needs_mb, free_mb)  # type: ignore[assignment]
        loading._budget_pools = {}
        return loading

    def test_a_model_that_cannot_fit_is_refused_before_it_is_started(self):
        with self.assertRaisesRegex(CouldNotLoadError, "needs about 9.8 GB"):
            self.loading(10000.0, 2000.0)._refuse_if_still_full(Shape.of("big", {}))

    def test_without_numbers_it_goes_ahead(self):
        self.loading(10000.0, 0.0)._refuse_if_still_full(Shape.of("big", {}))


def one_model_of_two_places():
    """The decisions alone, with one model of two places loaded."""
    planner = Planner(lambda shape: 2, lambda shape, loaded: [item["shape"] for item in loaded])
    return planner, QueueState(loaded={"a": 0})


def waiting(shape, still_wanted=None, number=1):
    return Waiting(number=number, shape=shape, arrived=0.0, still_wanted=still_wanted)


class PartsTests(unittest.TestCase):
    def test_a_failed_request_is_woken_with_its_reason(self):
        entry = waiting("a")
        entry.fail(AbandonedError("stopped"))
        self.assertTrue(entry.admitted.is_set())
        self.assertEqual(str(entry.failure), "stopped")

    def test_release_wakes_everyone_let_in(self):
        entries = [waiting("a", number=n) for n in range(3)]
        release(entries)
        self.assertTrue(all(entry.admitted.is_set() for entry in entries))

    def test_a_withdrawal_carries_its_reason_and_is_an_abandonment(self):
        reason = ValueError("the card is reserved")
        withdrawn = AdmissionWithdrawnError(reason)
        self.assertIs(withdrawn.reason, reason)
        self.assertIsInstance(withdrawn, AbandonedError)


class PlanningTests(unittest.TestCase):
    def setUp(self):
        self.planner, self.state = one_model_of_two_places()

    def test_requests_for_the_loaded_model_go_in_while_places_last(self):
        self.state.queue = [waiting("a", number=n) for n in range(3)]
        ready, switch = self.planner.plan(self.state)
        self.assertEqual((len(ready), switch, len(self.state.queue)), (2, None, 1))

    def test_another_model_is_a_switch_for_the_run_at_the_front(self):
        self.state.queue = [waiting("b", number=1), waiting("b", number=2),
                            waiting("a", number=3)]
        _, switch = self.planner.plan(self.state)
        assert switch is not None
        shape, victims, photograph, _ = switch
        self.assertEqual((shape, victims, len(photograph)), ("b", ("a",), 2))
        self.assertTrue(self.state.switching)

    def test_a_request_withdrawn_at_the_head_fails_with_its_reason(self):
        reason = ValueError("reserved")

        def withdrawn():
            raise AdmissionWithdrawnError(reason)
        entry = waiting("b", withdrawn)
        self.assertTrue(Planner.gone(entry))
        assert isinstance(entry.failure, AdmissionWithdrawnError)
        self.assertIs(entry.failure.reason, reason)

    def test_a_question_that_cannot_be_answered_serves_the_request(self):
        def broken():
            raise RuntimeError("no idea")
        self.assertFalse(Planner.gone(waiting("b", broken)))

    def test_a_load_that_failed_fails_its_whole_run(self):
        photograph = [waiting("b", number=1), waiting("b", number=2)]
        ready = self.planner.settle(self.state, ("b", ("a",), photograph, 0),
                                    CouldNotLoadError("no"))
        self.assertEqual(ready, [])
        self.assertTrue(all(isinstance(entry.failure, CouldNotLoadError) for entry in photograph))
        self.assertNotIn("a", self.state.loaded)

    def test_straight_through_only_when_nobody_waits_and_a_place_is_free(self):
        self.assertTrue(self.planner.admits_now(self.state, "a"))
        self.assertFalse(self.planner.admits_now(self.state, "b"))
        self.state.queue = [waiting("a")]
        self.assertFalse(self.planner.admits_now(self.state, "a"))
