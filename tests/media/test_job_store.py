"""Media jobs on disk: private files, written whole, recovered after a restart."""

import json
import stat
import tempfile
import unittest
from pathlib import Path

from ai_lab.media.job_store import JobStore, record_failure


class JobStoreTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp()) / "media-jobs"
        self.store = JobStore(self.root)

    def test_folders_and_files_are_private(self):
        self.store.save({"id": "a", "status": "queued"})
        self.assertEqual(stat.S_IMODE(self.root.stat().st_mode), 0o700)
        record = self.root / "jobs" / "a.json"
        self.assertEqual(stat.S_IMODE(record.stat().st_mode), 0o600)

    def test_a_result_comes_back_as_saved(self):
        self.store.save_result("a", {"data": [1, 2]})
        self.assertEqual(self.store.result("a"), {"data": [1, 2]})

    def test_recovery_fails_unfinished_jobs_and_drops_unreadable_ones(self):
        self.store.save({"id": "done", "status": "succeeded", "updated_at": 1})
        self.store.save({"id": "busy", "status": "running", "updated_at": 1})
        (self.root / "jobs" / "broken.json").write_text("{not json")
        (self.root / "jobs" / "other.json").write_text(json.dumps({"id": "mismatch"}))
        jobs = self.store.recover()
        self.assertEqual(sorted(jobs), ["busy", "done"])
        self.assertEqual(jobs["busy"]["status"], "failed")
        self.assertFalse((self.root / "jobs" / "broken.json").exists())

    def test_only_finished_jobs_older_than_the_cutoff_expire(self):
        jobs = {"old": {"status": "failed", "updated_at": 1},
                "new": {"status": "failed", "updated_at": 100},
                "busy": {"status": "running", "updated_at": 1}}
        for job_id, job in jobs.items():
            self.store.save({"id": job_id, **job})
        self.store.expire(jobs, cutoff=50)
        self.assertEqual(sorted(jobs), ["busy", "new"])
        self.assertFalse((self.root / "jobs" / "old.json").exists())


class FailureTests(unittest.TestCase):
    def test_a_cancelled_job_says_so_and_a_failed_one_its_error(self):
        cancelled: dict[str, str] = {}
        failed: dict[str, str] = {}
        record_failure(cancelled, True, RuntimeError("ignored"))
        record_failure(failed, False, RuntimeError("engine fell over"))
        self.assertEqual(cancelled, {"status": "cancelled", "error": "cancelled"})
        self.assertEqual(failed, {"status": "failed", "error": "engine fell over"})
