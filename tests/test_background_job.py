"""One background task at a time for one engine, and nothing switched under it."""

import threading
import unittest

from ai_lab.builds.background_job import BackgroundJob


class BackgroundJobTests(unittest.TestCase):
    def setUp(self):
        self.job = BackgroundJob("build-test", "a build")
        self.started, self.release = threading.Event(), threading.Event()

    def hold(self):
        self.started.set()
        self.release.wait(5)
        self.job.succeeded()

    def test_a_task_runs_once_and_ends_done(self):
        self.job.start(self.hold)
        self.assertTrue(self.started.wait(5))
        self.assertEqual(self.job.state, "running")
        with self.assertRaisesRegex(ValueError, "already running"):
            self.job.start(self.hold)
        self.release.set()
        assert self.job.thread is not None
        self.job.thread.join(5)
        self.assertEqual(self.job.state, "done")

    def test_a_change_is_refused_while_a_task_runs(self):
        self.job.start(self.hold)
        self.assertTrue(self.started.wait(5))
        with self.assertRaisesRegex(ValueError, "Cannot switch versions while a build is running"), \
                self.job.idle("switch versions"):
            pass
        self.release.set()

    def test_a_task_waits_for_a_change_in_progress(self):
        with self.job.idle("remove a version"), \
                self.assertRaisesRegex(ValueError, "being switched or removed"):
            self.job.start(self.hold)
        self.job.start(self.hold)        # the change is over: it starts
        self.release.set()

    def test_a_failure_is_kept_with_its_reason_and_the_lines_are_cleared(self):
        self.job.say("an old line")
        self.assertEqual(self.job.failed(RuntimeError("cmake exited 2")), "cmake exited 2")
        self.assertEqual((self.job.state, self.job.error), ("failed", "cmake exited 2"))
        self.job.start(lambda: None)
        assert self.job.thread is not None
        self.job.thread.join(5)
        self.assertEqual((self.job.error, self.job.log()), ("", []))

    def test_a_check_may_refuse_the_start(self):
        def refuse():
            raise ValueError("No git checkout")
        with self.assertRaisesRegex(ValueError, "No git checkout"):
            self.job.start(self.hold, check=refuse)
        self.assertEqual(self.job.state, "idle")
