"""The durable record of every model move, so a crash is reported, not lost.

`MoveJobStore` is held by `ModelStorageService`. One JSON file per move under
the host's state directory, rewritten atomically at every phase (`pending`,
`copying`, `verifying`, `publishing`, then `completed`, `failed` or
`cancelled`). Reading the file back is also how a cancellation asked for by
another request reaches the move in progress. The files are the only state,
each rewritten in one step, so the store needs no lock.
"""

from __future__ import annotations

import json
import os
import shutil
import time
import uuid
from collections.abc import Callable
from pathlib import Path

from ..hosts.base import Host
from .move_files import MovePlan

UNFINISHED_MOVE_STATUSES = ("pending", "copying", "verifying", "publishing", "cancelling")


class MoveCancelledError(Exception):
    """An operator cancelled a move; the job records this separately."""


class MoveJobStore:
    """The move job files under the host's state directory."""

    UNFINISHED_MOVE_STATUSES = UNFINISHED_MOVE_STATUSES

    def __init__(self, host: Host, on_change: Callable[[str], None]) -> None:
        self.host = host
        self._changed = on_change

    def open(self, plan: MovePlan) -> dict:
        """Write the durable job record before any byte is copied."""
        job_id = uuid.uuid4().hex
        # A sibling of the format directories, not inside one — the catalog
        # only scans each configured repository's own path (see
        # `Catalog.scan`), so a directory here never appears as a half-copied
        # model in the Library while a move is in flight or after one fails.
        staging_root = Path(plan.target_root.path) / ".ai-lab-staging"
        staging_root.mkdir(parents=True, exist_ok=True)
        os.chmod(staging_root, 0o700)
        job = {"id": job_id, "model_id": plan.model_id,
               "target_model_id": plan.model_id.replace(
                   plan.source_repository.id, plan.target_repository.id, 1),
               "source_tier": plan.source_repository.root_id,
               "target_tier": plan.target_root.id,
               "bytes": plan.model.size_bytes, "files": len(plan.sources),
               "staging": str(staging_root / job_id), "status": "pending",
               "error": "", "started_at": time.time(),
               "updated_at": time.time()}
        self.write(job)
        return job

    def finish(self, job: dict, status: str, error: str = "") -> None:
        """Record how the move ended, with the error when there is one."""
        if error:
            job["error"] = error
        job["status"] = status
        job["updated_at"] = time.time()
        self.write(job)

    def _move_job_dir(self) -> Path:
        directory = self.host.state_dir() / "moves"
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def _job_path(self, job_id: str) -> Path:
        return self._move_job_dir() / f"{job_id}.json"

    def write(self, job: dict) -> None:
        """Save the job record in one step (written beside, then renamed over)."""
        path = self._job_path(job["id"])
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(job, sort_keys=True))
        temporary.replace(path)
        self._changed("models")

    def read(self, job_id: str) -> dict:
        """The job record as it is on disk now."""
        return json.loads(self._job_path(job_id).read_text())

    def phase(self, job: dict, status: str) -> None:
        """Record that the move entered `status`."""
        job["status"] = status
        job["updated_at"] = time.time()
        self.write(job)

    def check_cancelled(self, job: dict) -> None:
        """Whether someone asked for this move to stop.

        Reading the job file back, rather than an in-memory flag, is what
        lets `cancel_move` be called from a different request than the one
        running the move.
        """
        try:
            current = self.read(job["id"])
        except FileNotFoundError:
            return
        if current.get("status") == "cancelling":
            raise MoveCancelledError()

    def all(self) -> list[dict]:
        """Every move job on record, most recently updated first.

        Read after a restart to find one that never reached `completed` — a
        resumable failure to report, not a move that vanished without a
        trace.
        """
        directory = self._move_job_dir()
        jobs = []
        for path in directory.glob("*.json"):
            try:
                jobs.append(json.loads(path.read_text()))
            except (json.JSONDecodeError, OSError):
                continue
        jobs.sort(key=lambda item: item.get("updated_at", 0), reverse=True)
        return jobs

    def recover(self) -> list[dict]:
        """Sweep move jobs left mid-flight by a process that did not exit cleanly.

        Called once at startup, before anything else touches the move job
        directory. A job still marked `copying` etc. has no thread behind it
        any more — the process that was running it is the one that just
        restarted — so it is explicitly marked failed rather than left to
        claim, forever, that a move is still in progress. Its staged bytes,
        which live outside the scanned tree (see `ModelStorageService.move_model`), are removed:
        recovery is one of the four cases the spec names for cleaning up
        temporary files, the other three being success, cancellation and
        timeout.
        """
        recovered = []
        for job in self.all():
            if job.get("status") not in self.UNFINISHED_MOVE_STATUSES:
                continue
            staging = job.get("staging")
            if staging:
                shutil.rmtree(staging, ignore_errors=True)
            job["status"] = "failed"
            job["error"] = "Interrupted by a service restart"
            job["updated_at"] = time.time()
            self.write(job)
            recovered.append(job)
        return recovered

    def cancel(self, job_id: str) -> dict:
        """Ask an in-progress move to stop at the next file or phase boundary.

        Does not touch the staged copy — the files already verified stay on
        disk, so the same job can be inspected or cleaned up rather than the
        work simply disappearing.
        """
        job = self.read(job_id)
        if job["status"] in ("completed", "failed", "cancelled"):
            return job
        job["status"] = "cancelling"
        job["updated_at"] = time.time()
        self.write(job)
        return job
