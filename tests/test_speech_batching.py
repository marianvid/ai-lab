import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor

from ai_lab.speech.batching import BatchQueue


class BatchQueueTests(unittest.TestCase):
    def test_requests_arriving_together_share_one_group(self):
        groups = []
        queue = BatchQueue(lambda items: groups.append(list(items)) or
                           [item * 10 for item in items], 8, 0.3)
        with ThreadPoolExecutor(4) as pool:
            results = list(pool.map(queue.submit, [1, 2, 3, 4]))
        self.assertEqual(results, [10, 20, 30, 40])
        self.assertEqual(len(groups), 1)
        self.assertEqual(sorted(groups[0]), [1, 2, 3, 4])

    def test_groups_never_exceed_the_batch_size(self):
        sizes = []
        release = threading.Event()

        def run(items):
            release.wait(2)
            sizes.append(len(items))
            return list(items)

        queue = BatchQueue(run, 3, 0.2)
        with ThreadPoolExecutor(7) as pool:
            futures = [pool.submit(queue.submit, n) for n in range(7)]
            time.sleep(0.3)
            release.set()
            self.assertEqual(sorted(f.result(5) for f in futures), list(range(7)))
        self.assertTrue(all(size <= 3 for size in sizes))
        self.assertEqual(sum(sizes), 7)

    def test_a_lone_request_waits_only_for_the_window(self):
        queue = BatchQueue(lambda items: list(items), 8, 0.05)
        started = time.monotonic()
        self.assertEqual(queue.submit('x'), 'x')
        self.assertLess(time.monotonic() - started, 1)

    def test_failures_reach_the_right_waiters(self):
        def run(items):
            return [ValueError(item) if item == 'bad' else item for item in items]

        queue = BatchQueue(run, 8, 0.2)
        with ThreadPoolExecutor(2) as pool:
            good = pool.submit(queue.submit, 'good')
            bad = pool.submit(queue.submit, 'bad')
            self.assertEqual(good.result(5), 'good')
            with self.assertRaises(ValueError):
                bad.result(5)

        def crash(items):
            raise RuntimeError('model died')

        broken = BatchQueue(crash, 8, 0)
        with self.assertRaises(RuntimeError):
            broken.submit('any')
        # The worker survives a crashed group.
        with self.assertRaises(RuntimeError):
            broken.submit('again')

    def test_limits_are_checked(self):
        for size, window in ((0, 0.1), (65, 0.1), (4, -1), (4, 6)):
            with self.assertRaises(ValueError):
                BatchQueue(lambda items: items, size, window)


if __name__ == '__main__':
    unittest.main()
