"""Private, atomic metadata and result storage for generated media."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

FINAL = frozenset({"succeeded", "failed", "cancelled"})


class JobStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.jobs_dir = self.root / "jobs"
        self.results_dir = self.root / "results"
        for directory in (self.root, self.jobs_dir, self.results_dir):
            directory.mkdir(parents=True, exist_ok=True)
            directory.chmod(0o700)

    @staticmethod
    def _atomic(path: Path, data: bytes) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
        os.replace(temporary, path)

    def save(self, job: dict) -> None:
        self._atomic(self.jobs_dir / f"{job['id']}.json",
                     json.dumps(job, separators=(",", ":")).encode())

    def result(self, job_id: str) -> dict:
        return json.loads((self.results_dir / f"{job_id}.json").read_text())

    def save_result(self, job_id: str, result: dict) -> None:
        self._atomic(self.results_dir / f"{job_id}.json",
                     json.dumps(result, separators=(",", ":")).encode())

    def recover(self) -> dict[str, dict]:
        jobs = {}
        for path in self.jobs_dir.glob("*.json"):
            try:
                job = json.loads(path.read_text())
                if job.get("id") != path.stem:
                    raise ValueError("Job ID does not match its file")
                if job.get("status") not in FINAL:
                    job["status"] = "failed"
                    job["error"] = "manager restarted before the job completed"
                    job["updated_at"] = time.time()
                    job["completed_at"] = job["updated_at"]
                    self.save(job)
                jobs[job["id"]] = job
            except (OSError, ValueError, KeyError):
                path.unlink(missing_ok=True)
        return jobs

    def expire(self, jobs: dict[str, dict], cutoff: float) -> None:
        for job_id, job in list(jobs.items()):
            if job["status"] in FINAL and job["updated_at"] < cutoff:
                jobs.pop(job_id)
                (self.jobs_dir / f"{job_id}.json").unlink(missing_ok=True)
                (self.results_dir / f"{job_id}.json").unlink(missing_ok=True)
