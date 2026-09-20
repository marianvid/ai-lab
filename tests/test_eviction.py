import unittest

from ai_lab.eviction import EvictionPlanner


class EvictionPlannerTests(unittest.TestCase):
    def setUp(self):
        self.planner = EvictionPlanner()
        self.sizes = {'wanted': 7, 'idle-old': 3, 'idle-new': 3, 'busy': 4}

    def choose(self, free=1, capacity=11):
        loaded = [
            {'shape': 'busy', 'in_flight': 1, 'last_used': 0},
            {'shape': 'idle-new', 'in_flight': 0, 'last_used': 2},
            {'shape': 'idle-old', 'in_flight': 0, 'last_used': 1},
        ]
        return self.planner.choose('wanted', loaded,
                                   needs_mb=self.sizes.get, free_mb=free,
                                   capacity_mb=capacity)

    def test_idle_victims_are_selected_before_busy(self):
        self.assertEqual(self.choose(), ['idle-old', 'idle-new'])

    def test_model_that_fits_needs_no_eviction(self):
        self.assertEqual(self.choose(free=8), [])

    def test_impossible_request_does_not_evict_anything(self):
        self.assertIsNone(self.choose(capacity=6))

    def test_unknown_size_clears_the_card_conservatively(self):
        self.sizes['wanted'] = 0
        self.assertEqual(self.choose(), ['idle-old', 'idle-new', 'busy'])


if __name__ == '__main__':
    unittest.main()
