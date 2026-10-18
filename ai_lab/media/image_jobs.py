"""Durable, named-profile image jobs owned by AI-Lab."""

from __future__ import annotations

import base64
import contextlib
import copy
import hashlib
import json
import os
import queue
import shutil
import threading
import time
import urllib.request
import uuid
from pathlib import Path

from ..multipart import MultipartBody
from ..network import LOOPBACK
from ..types import ChangeEvent, Task
from .job_store import record_failure
from ..uploads import validate_image
from .image_workflows import ALLOWED, read_request, render

FINAL = frozenset({"succeeded", "failed", "cancelled"})
# How long a waiting client is kept, how often it looks, how long a result lives.
CLIENT_WAIT_S = 1800
WAIT_POLL_S = 0.1
RESULT_TTL_S = 86400
# The settings a job records as the ones it was run with.
EFFECTIVE = ("seed", "width", "height", "steps", "cfg_scale", "n")


class ImageJobs:
    """Image jobs by named profile: queued, run one at a time, kept on disk.

    A job survives a client that stops waiting (`async`), and its result is
    kept for `result_ttl_s`. A manager restart marks unfinished jobs failed
    rather than leaving them "running" for ever.
    """

    def __init__(self, gateway, settings: dict, state_root: Path, bus=None) -> None:
        self.gateway = gateway
        self.settings = settings or {}
        self.bus = bus
        self.root = Path(self.settings.get("state_dir") or state_root / "image-jobs")
        self.jobs_dir = self.root / "jobs"
        self.temp_dir = self.root / "temporary"
        self.results_dir = self.root / "results"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.jobs: dict[str, dict] = {}
        self.payloads: dict[str, dict] = {}
        self.cancelled: set[str] = set()
        self.pending: queue.Queue[str] = queue.Queue()
        self.lock = threading.RLock()
        self.running: str | None = None
        self.running_port: int | None = None
        self._recover()
        threading.Thread(target=self._work, daemon=True,
                         name="image-jobs").start()

    def profiles(self) -> list[dict]:
        """The named profiles a client may ask for."""
        return [{"id": key, "task": value.get("task", "generation"),
                 "model": value.get("model", ""),
                 "description": value.get("description", "")}
                for key, value in sorted(self.settings.get("profiles", {}).items())]

    def list(self) -> list[dict]:
        """Every job, newest first, without the images themselves."""
        with self.lock:
            return [self._summary(job) for job in sorted(
                self.jobs.values(), key=lambda item: item["created_at"], reverse=True)]

    def get(self, job_id: str) -> dict:
        """One job, with its images once it succeeded."""
        with self.lock:
            if job_id not in self.jobs:
                raise KeyError(f"Unknown image job: {job_id}")
            return self._public(self.jobs[job_id])

    def submit(self, body: dict | MultipartBody, task: Task) -> dict:
        """Queue one image job; wait for it unless the request says `async`."""
        payload, image = read_request(body)
        profile_id, profile = self._profile(payload, task, image)
        rendered = render(profile, payload, Path(self.settings.get("workflow_root", ".")))
        job = _new_job(profile_id, profile, task, payload, rendered)
        work = {"workflow": rendered, "image_base64":
                base64.b64encode(image).decode() if image else None}
        with self.lock:
            self.jobs[job["id"]] = job
            self.payloads[job["id"]] = work
            self._save(job)
            self.pending.put(job["id"])
        self._changed()
        if payload.get("async") is True:
            return self._public(job)
        return self._wait_for(job["id"], job["created_at"])

    def _profile(self, payload: dict, task: Task, image: bytes | None) -> tuple[str, dict]:
        """The named profile, checked against the request; the image checked too."""
        unknown = set(payload) - ALLOWED
        if unknown:
            raise ValueError("Unknown image fields: " + ", ".join(sorted(unknown)))
        profile_id = str(payload.get("profile", ""))
        profile = self.settings.get("profiles", {}).get(profile_id)
        if not profile:
            raise ValueError(f"Unknown image profile: {profile_id}")
        expected = "edit" if task is Task.IMAGE_EDIT else "generation"
        if profile.get("task", "generation") != expected:
            raise ValueError(f"Profile {profile_id} does not support {expected}")
        if task is Task.IMAGE_EDIT and not image:
            raise ValueError("image edits require an image file")
        if image:
            gateway = self.gateway
            validate_image(image, max_bytes=gateway.max_upload_bytes,
                           max_pixels=gateway.max_upload_pixels,
                           max_dimension=gateway.max_upload_dimension)
        return profile_id, profile

    def _wait_for(self, job_id: str, created: float) -> dict:
        """The finished job's images, within `client_wait_s`; its error if it failed."""
        deadline = time.monotonic() + float(self.settings.get("client_wait_s", CLIENT_WAIT_S))
        while time.monotonic() < deadline:
            current = self.get(job_id)
            if current["status"] in FINAL:
                if current["status"] != "succeeded":
                    raise RuntimeError(current["error"] or current["status"])
                return {"created": int(created), "job_id": job_id,
                        "data": current["result"]["data"]}
            time.sleep(WAIT_POLL_S)
        raise TimeoutError(f"Image job {job_id} is still running")

    def cancel(self, job_id: str) -> dict:
        """Cancel a job: at once while it waits, through the engine while it runs."""
        with self.lock:
            job = self.jobs.get(job_id)
            if not job:
                raise KeyError(f"Unknown image job: {job_id}")
            if job["status"] in FINAL:
                return self._public(job)
            self.cancelled.add(job_id)
            job["status"] = "cancelling" if self.running == job_id else "cancelled"
            job["updated_at"] = time.time()
            self._save(job)
            running = self.running == job_id
        if running:
            self._interrupt(job["model"])
        self._changed()
        return self.get(job_id)

    def cleanup(self) -> None:
        """Forget finished jobs older than `result_ttl_s`, and their files."""
        ttl = float(self.settings.get("result_ttl_s", RESULT_TTL_S))
        cutoff = time.time() - ttl
        with self.lock:
            expired = [key for key, job in self.jobs.items()
                       if job["status"] in FINAL and job["updated_at"] < cutoff]
            for key in expired:
                self.jobs.pop(key, None)
                self.payloads.pop(key, None)
                (self.jobs_dir / f"{key}.json").unlink(missing_ok=True)
                shutil.rmtree(self.temp_dir / key, ignore_errors=True)
                shutil.rmtree(self.results_dir / key, ignore_errors=True)

    def _work(self) -> None:
        """The worker thread: one job at a time, in the order they were queued."""
        while True:
            job_id = self.pending.get()
            started = self._start(job_id)
            if started is None:
                continue
            job, work = started
            self._changed()
            try:
                result = self._call(job, work)
                with self.lock:
                    if job_id in self.cancelled:
                        raise RuntimeError("cancelled")
                    job["status"] = "succeeded"
                    job["result"] = self._store_result(job_id, result)
                    job["error"] = ""
            except Exception as error:
                with self.lock:
                    record_failure(job, job_id in self.cancelled, error)
            finally:
                self._finish(job)

    def _start(self, job_id: str) -> tuple[dict, dict] | None:
        """Mark the job running; None when it is gone or was cancelled while queued."""
        with self.lock:
            job = self.jobs.get(job_id)
            if not job or job_id in self.cancelled:
                return None
            self.running = job_id
            job["status"] = "running"
            job["updated_at"] = time.time()
            job["started_at"] = job["updated_at"]
            self._save(job)
            return job, self.payloads[job_id]

    def _call(self, job: dict, work: dict) -> dict:
        """The engine's answer to the job, holding the card for the whole of it."""
        path = ("/v1/images/edits" if job["task"] == Task.IMAGE_EDIT.value
                else "/v1/images/generations")
        with self.gateway.acquire(job["model"], shape=path) as lease:
            with self.lock:
                self.running_port = lease.port
            request = urllib.request.Request(
                f"http://{LOOPBACK}:{lease.port}{path}",
                data=json.dumps(work).encode(), method="POST",
                headers={"Content-Type": "application/json"})
            first, _ = self.gateway.timeouts_for(Task(job["task"]))
            with urllib.request.urlopen(request, timeout=first) as response:
                return json.loads(response.read())

    def _finish(self, job: dict) -> None:
        """Record when and how long, forget the job's input, tidy up."""
        with self.lock:
            job["updated_at"] = time.time()
            job["completed_at"] = job["updated_at"]
            if job.get("started_at"):
                job["duration_ms"] = int((job["completed_at"] - job["started_at"]) * 1000)
            self._save(job)
            self.payloads.pop(job["id"], None)
            self.running = None
            self.running_port = None
        shutil.rmtree(self.temp_dir / job["id"], ignore_errors=True)
        self.cleanup()
        self._changed()

    def _render(self, profile: dict, values: dict) -> dict:
        """The profile's graph for these values (see `workflows.render`)."""
        return render(profile, values, Path(self.settings.get("workflow_root", ".")))

    def _interrupt(self, model: str) -> None:
        """Ask the engine running the job to stop it; nothing if it cannot be reached."""
        with self.lock:
            port = self.running_port
        if port is None:
            return
        request = urllib.request.Request(
            f"http://{LOOPBACK}:{port}/api/jobs/current/cancel",
            data=b"{}", method="POST", headers={"Content-Type": "application/json"})
        with contextlib.suppress(Exception):
            urllib.request.urlopen(request, timeout=5).close()

    def _recover(self) -> None:
        for path in self.jobs_dir.glob("*.json"):
            try:
                job = json.loads(path.read_text())
                if job.get("status") not in FINAL:
                    job["status"] = "failed"
                    job["error"] = "manager restarted before the job completed"
                    job["updated_at"] = time.time()
                    self._save(job)
                self.jobs[job["id"]] = job
            except Exception:
                path.unlink(missing_ok=True)
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.cleanup()

    def _save(self, job: dict) -> None:
        path = self.jobs_dir / f"{job['id']}.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(job, separators=(",", ":")))
        os.replace(temporary, path)

    @staticmethod
    def _descriptor(index: int, item: dict) -> dict:
        return {"id": f"image-{index}",
                "mime_type": item.get("mime_type", "image/png")}

    def _store_result(self, job_id: str, result: dict) -> dict:
        directory = self.results_dir / job_id
        directory.mkdir(parents=True, exist_ok=True)
        descriptors = []
        for index, item in enumerate(result.get("data", [])):
            raw = base64.b64decode(item["b64_json"])
            target = directory / f"image-{index}.bin"
            temporary = target.with_suffix(".tmp")
            temporary.write_bytes(raw)
            os.replace(temporary, target)
            descriptors.append(self._descriptor(index, item))
        if not descriptors:
            raise RuntimeError("image engine returned no images")
        return {"data": descriptors}

    def _public(self, job: dict) -> dict:
        public = copy.deepcopy(job)
        if job.get("status") == "succeeded" and job.get("result"):
            data = []
            for item in job["result"].get("data", []):
                path = self.results_dir / job["id"] / f"{item['id']}.bin"
                if path.is_file():
                    data.append({**item, "b64_json":
                                 base64.b64encode(path.read_bytes()).decode()})
            public["result"] = {"data": data}
        return public

    @staticmethod
    def _summary(job: dict) -> dict:
        """A job row without the potentially multi-megabyte image payload."""
        public = copy.deepcopy(job)
        result = public.get("result")
        if result:
            public["result"] = {"data": [
                {key: value for key, value in item.items() if key != "b64_json"}
                for item in result.get("data", [])]}
        return public

    def _changed(self) -> None:
        if self.bus is not None:
            self.bus.publish(ChangeEvent(topic="image-jobs"))


def _new_job(profile_id: str, profile: dict, task: Task, payload: dict,
             rendered: dict) -> dict:
    """The record of a job just queued: what was asked and which graph will run."""
    model = str(profile.get("model") or payload.get("model") or "")
    if not model:
        raise ValueError(f"Profile {profile_id} has no model")
    now = time.time()
    graph_hash = hashlib.sha256(json.dumps(
        rendered, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"id": uuid.uuid4().hex, "profile": profile_id,
            "profile_version": str(profile.get("version", "1")),
            "workflow_sha256": graph_hash, "model": model,
            "task": task.value, "status": "queued", "created_at": now,
            "updated_at": now, "started_at": None, "completed_at": None,
            "duration_ms": None,
            "effective_settings": {key: payload[key] for key in EFFECTIVE if key in payload},
            "error": "", "result": None}
