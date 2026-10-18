"""Durable, cancellable jobs for music and speech generation."""

from __future__ import annotations

import contextlib
import json
import queue
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

from ..config_policy import MediaPolicy
from ..engines.base import MUSIC_PATHS, SPEECH_PATHS
from ..network import LOOPBACK
from ..types import ChangeEvent, Task
from .job_store import FINAL, JobStore, record_failure

PATHS = {
    Task.MUSIC_GENERATION.value: MUSIC_PATHS[0],
    Task.SPEECH_SYNTHESIS.value: SPEECH_PATHS[0],
}


def engine_refusal(error: urllib.error.HTTPError) -> str:
    """The engine's own reason for refusing a job, not just its status line.

    Engines answer `{"error": {"message": ...}}`. Keeping that message is what
    tells the caller which field to fix; "HTTP Error 400: Bad Request" alone
    does not.
    """
    try:
        body = json.loads(error.read(65536) or b"{}")
        message = body.get("error", {}).get("message") or ""
    except Exception:
        message = ""
    status = f"HTTP {error.code}"
    return f"{status}: {message}" if message else f"{status}: {error.reason}"


class MediaJobs:
    """Music and speech jobs: queued, run one at a time, kept on disk, cancellable."""

    def __init__(self, gateway, state_root: Path, bus=None,
                 *, settings: MediaPolicy | dict | None = None) -> None:
        self.gateway = gateway
        self.bus = bus
        policy = (settings if isinstance(settings, MediaPolicy)
                  else MediaPolicy.from_mapping(settings))
        self.ttl_s = policy.result_ttl_s
        self.max_queue = policy.max_queue
        self.max_input_bytes = policy.max_input_bytes
        self.max_result_bytes = policy.max_result_bytes
        self.store = JobStore(Path(state_root) / "media-jobs")
        self.lock = threading.RLock()
        self.jobs = self.store.recover()
        self.payloads: dict[str, dict] = {}
        self.pending: queue.Queue[str] = queue.Queue()
        self.cancelled: set[str] = set()
        self.running: str | None = None
        self.running_port: int | None = None
        self.cleanup()
        threading.Thread(target=self._work, name="media-jobs", daemon=True).start()

    def submit(self, body: dict) -> dict:
        """Queue one job for `model`; its result is fetched later by id."""
        if not isinstance(body, dict) or set(body) != {"model", "task", "input"}:
            raise ValueError("Media job needs model, task and input")
        task, model, payload = body["task"], body["model"], body["input"]
        if task not in PATHS:
            raise ValueError(f"Unsupported media task: {task}")
        if not isinstance(model, str) or not model:
            raise ValueError("Media job needs a model name")
        if not isinstance(payload, dict):
            raise ValueError("Media input must be an object")
        if len(json.dumps(payload).encode()) > self.max_input_bytes:
            raise ValueError("Media input exceeds the configured limit")
        # Resolve before queuing: a misspelt model is a request error, not a
        # durable job that fails later. Shape support is checked at acquisition.
        self.gateway.resolve(model)
        now = time.time()
        job_id = uuid.uuid4().hex
        job = {"id": job_id, "model": model, "task": task,
               "status": "queued", "created_at": now, "updated_at": now,
               "started_at": None, "completed_at": None,
               "duration_ms": None, "error": "", "has_result": False}
        with self.lock:
            if sum(item["status"] not in FINAL for item in self.jobs.values()) >= self.max_queue:
                raise ValueError("Too many media jobs are waiting")
            self.jobs[job_id] = job
            self.payloads[job_id] = payload
            self.store.save(job)
            self.pending.put(job_id)
        self._changed()
        return dict(job)

    def list(self) -> list[dict]:
        """Every job, newest first."""
        with self.lock:
            return [dict(item) for item in sorted(self.jobs.values(),
                    key=lambda job: job["created_at"], reverse=True)]

    def get(self, job_id: str) -> dict:
        """One job, with its result once it succeeded."""
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None:
                raise KeyError(f"Unknown media job: {job_id}")
            answer = dict(job)
            if job["status"] == "succeeded":
                answer["result"] = self.store.result(job_id)
            return answer

    def cancel(self, job_id: str) -> dict:
        """Cancel a job: at once while it waits, through the engine while it runs."""
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None:
                raise KeyError(f"Unknown media job: {job_id}")
            if job["status"] in FINAL:
                return self.get(job_id)
            self.cancelled.add(job_id)
            running = self.running == job_id
            job["status"] = "cancelling" if running else "cancelled"
            job["updated_at"] = time.time()
            if not running:
                job["completed_at"] = job["updated_at"]
                self.payloads.pop(job_id, None)
            self.store.save(job)
            port = self.running_port if running else None
        if port is not None:
            self._interrupt(port)
        self._changed()
        return self.get(job_id)

    def cleanup(self) -> None:
        """Forget finished jobs older than the result lifetime."""
        with self.lock:
            self.store.expire(self.jobs, time.time() - self.ttl_s)

    def _work(self) -> None:
        """The worker thread: one job at a time, in the order they were queued."""
        while True:
            job_id = self.pending.get()
            started = self._start(job_id)
            if started is None:
                continue
            job, payload = started
            self._changed()
            try:
                result = self._call(job, payload)
                with self.lock:
                    if job_id not in self.cancelled:
                        self.store.save_result(job_id, result)
                        job["status"] = "succeeded"
                        job["has_result"] = True
            except Exception as error:
                with self.lock:
                    record_failure(job, job_id in self.cancelled, error)
            finally:
                self._finish(job)

    def _start(self, job_id: str) -> tuple[dict, dict] | None:
        """Mark the job running; None when it is gone or was cancelled while queued."""
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None or job_id in self.cancelled:
                return None
            self.running = job_id
            job["status"] = "running"
            job["started_at"] = job["updated_at"] = time.time()
            self.store.save(job)
            return job, self.payloads[job_id]

    def _call(self, job: dict, payload: dict) -> dict:
        """The engine's answer to the job, holding the card for the whole of it."""
        path = PATHS[job["task"]]
        with self.gateway.acquire(job["model"], shape=path) as lease:
            with self.lock:
                self.running_port = lease.port
                if job["id"] in self.cancelled:
                    raise RuntimeError("cancelled")
            outgoing = {**payload, "model": lease.model_name or job["model"]}
            request = urllib.request.Request(
                f"http://{LOOPBACK}:{lease.port}{path}",
                data=json.dumps(outgoing).encode(), method="POST",
                headers={"Content-Type": "application/json"})
            first, _ = self.gateway.timeouts_for(Task(job["task"]))
            try:
                with urllib.request.urlopen(request, timeout=first) as response:
                    raw = response.read(self.max_result_bytes + 1)
            except urllib.error.HTTPError as error:
                raise RuntimeError(engine_refusal(error)) from error
        if len(raw) > self.max_result_bytes:
            raise ValueError("Media result exceeds the configured limit")
        return json.loads(raw)

    def _finish(self, job: dict) -> None:
        """Record when and how long, forget the job's input, tidy up."""
        with self.lock:
            if job["id"] in self.cancelled:
                job["status"] = "cancelled"
                job["error"] = "cancelled"
            job["updated_at"] = job["completed_at"] = time.time()
            job["duration_ms"] = int((job["completed_at"] - job["started_at"]) * 1000)
            self.store.save(job)
            self.payloads.pop(job["id"], None)
            self.running = self.running_port = None
        self.cleanup()
        self._changed()

    @staticmethod
    def _interrupt(port: int) -> None:
        """Ask the engine to stop the job it runs; nothing if it cannot be reached.

        ComfyUI understands this route. Other engines may finish naturally;
        their lease stays held until they do, even after a cancellation.
        """
        request = urllib.request.Request(
            f"http://{LOOPBACK}:{port}/api/jobs/current/cancel",
            data=b"{}", method="POST",
            headers={"Content-Type": "application/json"})
        with contextlib.suppress(Exception):
            urllib.request.urlopen(request, timeout=5).close()

    def _changed(self) -> None:
        if self.bus is not None:
            self.bus.publish(ChangeEvent(topic="media-jobs"))
