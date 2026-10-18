"""The move job records: a cancellation asked for by one request reaches the move in another."""

import tempfile
import threading
import unittest
from pathlib import Path
from typing import Any, cast

from ai_lab.application.move_jobs import MoveCancelledError, MoveJobStore


class Host:
    """Just the state directory."""

    def __init__(self):
        self.root = Path(tempfile.mkdtemp())

    def state_dir(self):
        return self.root


def a_job(job_id="job1", status="copying"):
    return {"id": job_id, "status": status, "error": "", "updated_at": 0.0}


class CancellingTests(unittest.TestCase):
    def setUp(self):
        self.store = MoveJobStore(cast(Any, Host()), lambda _topic: None)

    def test_a_cancel_between_the_last_copy_and_verifying_stops_the_move(self):
        # The move thread holds its own copy of the job; the operator cancels
        # from another request while the last file is being checksummed.
        job = a_job()
        self.store.write(dict(job))
        self.store.cancel("job1")
        with self.assertRaises(MoveCancelledError):
            self.store.phase(job, "verifying")
        self.assertEqual(self.store.read("job1")["status"], "cancelling")

    def test_a_late_completion_does_not_hide_the_cancel_request(self):
        job = a_job(status="publishing")
        self.store.write(dict(job))
        self.store.cancel("job1")
        self.store.finish(job, "completed")
        record = self.store.read("job1")
        self.assertEqual(record["status"], "completed")
        self.assertIn("cancel", record["error"].lower())

    def test_a_cancelled_move_is_recorded_as_cancelled(self):
        job = a_job()
        self.store.write(dict(job))
        self.store.cancel("job1")
        self.store.finish(job, "cancelled", "Cancelled")
        self.assertEqual(self.store.read("job1")["status"], "cancelled")

    def test_two_writers_at_once_leave_a_readable_record(self):
        self.store.write(a_job())
        errors = []
        start = threading.Barrier(8)

        def writer(index):
            start.wait(5)
            try:
                for _ in range(50):
                    self.store.write(a_job(status=f"copying-{index}"))
            except OSError as error:  # noqa: PERF203 - recorded for the test
                errors.append(error)
        threads = [threading.Thread(target=writer, args=(n,)) for n in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(10)
        self.assertEqual(errors, [])
        self.assertTrue(self.store.read("job1")["status"].startswith("copying"))
