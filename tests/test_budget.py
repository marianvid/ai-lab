"""How much memory is available for models.

The rule these defend: being wrong optimistically is the expensive direction.
Refusing a model that would have fitted costs a sentence on screen; starting
one that does not fit costs the model that was already working.
"""

from __future__ import annotations

import unittest

from ai_lab import budget
from ai_lab.types import AcceleratorSnapshot


class FakeHost:
    def __init__(self, card=None, machine=(0.0, 0.0)):
        self._card = card
        self._machine = machine

    def accelerator(self, pid=None):
        if self._card is None:
            raise RuntimeError("no card")
        return self._card

    def system_memory(self):
        return self._machine


def card(total=32623.0, used=0.0, kind="dedicated", available=True):
    return AcceleratorSnapshot(available=available, name="test", kind="cuda",
                               memory_kind=kind, memory_used_mb=used,
                               memory_total_mb=total)


class ADedicatedCard(unittest.TestCase):
    """Two pools: the card, used whole, and the machine, which is shared."""

    def setUp(self):
        self.host = FakeHost(card=card(total=32623, used=29138),
                             machine=(12000.0, 49152.0))
        self.budget = budget.of(self.host, reserve_mb=8192)

    def test_it_reports_both_pools(self):
        self.assertEqual([pool.name for pool in self.budget.pools],
                         [budget.CARD, budget.MACHINE])
        self.assertFalse(self.budget.unified)

    def test_the_whole_card_is_for_models(self):
        # Nothing else on the machine wants it, so no reserve is taken.
        card_pool = self.budget.pool(budget.CARD)
        self.assertEqual(card_pool.reserve_mb, 0.0)
        self.assertEqual(card_pool.available_mb, 32623 - 29138)

    def test_the_machine_keeps_its_reserve(self):
        machine = self.budget.pool(budget.MACHINE)
        self.assertEqual(machine.free_mb, 49152 - 12000)
        self.assertEqual(machine.available_mb, 49152 - 12000 - 8192)

    def test_a_machine_already_past_its_reserve_offers_nothing(self):
        # Not a negative amount, which would add up to a smaller total
        # elsewhere and read as though there were room.
        tight = budget.of(FakeHost(card=card(), machine=(46000.0, 49152.0)),
                          reserve_mb=8192)
        self.assertEqual(tight.pool(budget.MACHINE).available_mb, 0.0)

    def test_what_our_own_models_hold_is_separate_from_what_is_used(self):
        # On a card they are the same thing; on the machine they are not,
        # because the rest of the machine is using it too.
        self.assertEqual(self.budget.pool(budget.CARD).used_by_models_mb, 29138)
        self.assertEqual(self.budget.pool(budget.MACHINE).used_by_models_mb, 0.0)


class UnifiedMemory(unittest.TestCase):
    """One pool. Counting it twice would double the machine."""

    def setUp(self):
        self.host = FakeHost(card=card(total=131072, used=21000, kind="unified"),
                             machine=(34600.0, 131072.0))
        self.budget = budget.of(self.host, reserve_mb=16384)

    def test_there_is_exactly_one_pool(self):
        self.assertTrue(self.budget.unified)
        self.assertEqual([pool.name for pool in self.budget.pools],
                         [budget.MACHINE])

    def test_free_comes_from_the_machine_not_from_our_own_models(self):
        # The card reading only knows what our engines hold. A laptop somebody
        # is also working on has a browser and an editor in that pool, and
        # trusting the card reading would offer memory that is not there.
        pool = self.budget.pools[0]
        self.assertEqual(pool.used_mb, 34600.0)
        self.assertEqual(pool.available_mb, 131072 - 34600 - 16384)

    def test_it_still_reports_what_our_models_hold(self):
        # Both numbers are useful and neither replaces the other.
        self.assertEqual(self.budget.pools[0].used_by_models_mb, 21000)

    def test_the_total_is_not_the_two_readings_added(self):
        self.assertEqual(self.budget.available_mb,
                         self.budget.pools[0].available_mb)
        self.assertEqual(self.budget.capacity_mb,
                         self.budget.pools[0].capacity_mb)


class WhenTheMachineWillNotAnswer(unittest.TestCase):
    def test_a_host_that_raises_reports_no_pools_rather_than_no_room(self):
        class Broken:
            def accelerator(self, pid=None):
                raise RuntimeError("nvidia-smi is not there")

            def system_memory(self):
                raise RuntimeError("no /proc")

        found = budget.of(Broken())
        self.assertEqual(found.pools, ())
        self.assertEqual(found.available_mb, 0.0)
        self.assertEqual(found.capacity_mb, 0.0)

    def test_a_card_that_is_not_available_is_not_a_pool(self):
        found = budget.of(FakeHost(card=card(available=False, total=0),
                                   machine=(1000.0, 8000.0)))
        self.assertEqual([pool.name for pool in found.pools], [budget.MACHINE])

    def test_a_machine_that_cannot_read_its_own_memory_still_offers_the_card(self):
        # Which is what Linux looks like if /proc/meminfo is unreadable.
        found = budget.of(FakeHost(card=card(total=32623, used=0),
                                   machine=(0.0, 0.0)))
        self.assertEqual([pool.name for pool in found.pools], [budget.CARD])
        self.assertEqual(found.available_mb, 32623)


if __name__ == "__main__":
    unittest.main()


class TwoDifferentQuestions(unittest.TestCase):
    """How big this machine is, and how much is free right now.

    Confusing the two would either refuse a model that fits or accept one that
    does not, and the settings page and the gateway page ask different ones.
    """

    def setUp(self):
        # A card holding a 29 GB model, so the two answers are far apart.
        self.budget = budget.of(
            FakeHost(card=card(total=32623, used=28946),
                     machine=(5157.0, 49152.0)), reserve_mb=8192)

    def test_capacity_does_not_move_when_a_model_loads(self):
        pool = self.budget.pool(budget.CARD)
        self.assertEqual(pool.capacity_mb, 32623, "the whole card, whatever holds it")
        self.assertEqual(pool.available_mb, 32623 - 28946)

    def test_the_reserve_comes_out_of_capacity_too(self):
        machine = self.budget.pool(budget.MACHINE)
        self.assertEqual(machine.capacity_mb, 49152 - 8192)
        self.assertEqual(machine.available_mb, 49152 - 5157 - 8192)

    def test_the_reserve_is_one_number_not_a_sum(self):
        # Only the machine's own memory carries one; a card is used whole. A
        # sum would double it on unified memory, where both readings are one
        # pool.
        self.assertEqual(self.budget.reserve_mb, 8192)
        unified = budget.of(
            FakeHost(card=card(total=131072, used=21000, kind="unified"),
                     machine=(34600.0, 131072.0)), reserve_mb=16384)
        self.assertEqual(unified.reserve_mb, 16384)
