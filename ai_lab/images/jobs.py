"""Durable, named-profile image jobs owned by AI-Lab."""

from __future__ import annotations

import base64
import copy
import json
import os
import queue
import shutil
import threading
import time
import urllib.request
import uuid
import hashlib
from pathlib import Path

from ..api.multipart import MultipartBody
from ..api.uploads import validate_image
from ..types import ChangeEvent, Task

FINAL = frozenset({"succeeded", "failed", "cancelled"})
ALLOWED = frozenset({"profile", "model", "prompt", "negative_prompt", "seed",
                     "width", "height", "steps", "cfg_scale", "n", "async",
                     "response_format"})


class ImageJobs:
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
        return [{"id": key, "task": value.get("task", "generation"),
                 "model": value.get("model", ""),
                 "description": value.get("description", "")}
                for key, value in sorted(self.settings.get("profiles", {}).items())]

    def list(self) -> list[dict]:
        with self.lock:
            return [self._summary(job) for job in sorted(
                self.jobs.values(), key=lambda item: item["created_at"], reverse=True)]

    def get(self, job_id: str) -> dict:
        with self.lock:
            if job_id not in self.jobs:
                raise KeyError(f"Unknown image job: {job_id}")
            return self._public(self.jobs[job_id])

    def submit(self, body: dict | MultipartBody, task: Task) -> dict:
        payload, image = self._input(body)
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
        rendered = self._render(profile, payload)
        model = str(profile.get("model") or payload.get("model") or "")
        if not model:
            raise ValueError(f"Profile {profile_id} has no model")
        job_id = uuid.uuid4().hex
        now = time.time()
        effective = {key: payload[key] for key in
                     ("seed", "width", "height", "steps", "cfg_scale", "n")
                     if key in payload}
        workflow_version = str(profile.get("version", "1"))
        graph_hash = hashlib.sha256(json.dumps(
            rendered, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        job = {"id": job_id, "profile": profile_id,
               "profile_version": workflow_version,
               "workflow_sha256": graph_hash, "model": model,
               "task": task.value, "status": "queued", "created_at": now,
               "updated_at": now, "started_at": None, "completed_at": None,
               "duration_ms": None, "effective_settings": effective,
               "error": "", "result": None}
        work = {"workflow": rendered, "image_base64":
                base64.b64encode(image).decode() if image else None}
        with self.lock:
            self.jobs[job_id] = job
            self.payloads[job_id] = work
            self._save(job)
            self.pending.put(job_id)
        self._changed()
        if payload.get("async") is True:
            return self._public(job)
        timeout = float(self.settings.get("client_wait_s", 1800))
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            current = self.get(job_id)
            if current["status"] in FINAL:
                if current["status"] != "succeeded":
                    raise RuntimeError(current["error"] or current["status"])
                return {"created": int(now), "job_id": job_id,
                        "data": current["result"]["data"]}
            time.sleep(.1)
        raise TimeoutError(f"Image job {job_id} is still running")

    def cancel(self, job_id: str) -> dict:
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
        ttl = float(self.settings.get("result_ttl_s", 86400))
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
        while True:
            job_id = self.pending.get()
            with self.lock:
                job = self.jobs.get(job_id)
                if not job or job_id in self.cancelled:
                    continue
                self.running = job_id
                job["status"] = "running"
                job["updated_at"] = time.time()
                job["started_at"] = job["updated_at"]
                self._save(job)
                work = self.payloads[job_id]
            self._changed()
            try:
                path = ("/v1/images/edits" if job["task"] == Task.IMAGE_EDIT.value
                        else "/v1/images/generations")
                with self.gateway.acquire(job["model"], shape=path) as lease:
                    with self.lock:
                        self.running_port = lease.port
                    request = urllib.request.Request(
                        f"http://127.0.0.1:{lease.port}{path}",
                        data=json.dumps(work).encode(), method="POST",
                        headers={"Content-Type": "application/json"})
                    first, _ = self.gateway.timeouts_for(Task(job["task"]))
                    with urllib.request.urlopen(request, timeout=first) as response:
                        result = json.loads(response.read())
                with self.lock:
                    if job_id in self.cancelled:
                        raise RuntimeError("cancelled")
                    job["status"] = "succeeded"
                    job["result"] = self._store_result(job_id, result)
                    job["error"] = ""
            except Exception as error:
                with self.lock:
                    job["status"] = "cancelled" if job_id in self.cancelled else "failed"
                    job["error"] = "cancelled" if job_id in self.cancelled else str(error)
            finally:
                with self.lock:
                    job["updated_at"] = time.time()
                    job["completed_at"] = job["updated_at"]
                    if job.get("started_at"):
                        job["duration_ms"] = int(
                            (job["completed_at"] - job["started_at"]) * 1000)
                    self._save(job)
                    self.payloads.pop(job_id, None)
                    self.running = None
                    self.running_port = None
                shutil.rmtree(self.temp_dir / job_id, ignore_errors=True)
                self.cleanup()
                self._changed()

    def _render(self, profile: dict, values: dict) -> dict:
        workflow_root = Path(self.settings.get("workflow_root", ".")).resolve()
        path = (workflow_root / str(profile.get("workflow", ""))).resolve()
        if workflow_root != path and workflow_root not in path.parents:
            raise ValueError("Profile workflow escapes workflow_root")
        graph = json.loads(path.read_text())
        if not isinstance(graph, dict) or not graph:
            raise ValueError("Profile workflow is not an API-format ComfyUI graph")
        mappings = profile.get("inputs", {})
        for name, mapping in mappings.items():
            if name == "image":
                value = "__AI_LAB_INPUT__"
            elif name not in values:
                continue
            else:
                value = values[name]
            if not isinstance(mapping, list) or len(mapping) != 2:
                raise ValueError(f"Invalid input mapping for {name}")
            node, key = str(mapping[0]), str(mapping[1])
            if node not in graph or "inputs" not in graph[node] or key not in graph[node]["inputs"]:
                raise ValueError(f"Workflow input {name} targets missing {node}.{key}")
            graph[node]["inputs"][key] = value
        for node in graph.values():
            if not isinstance(node, dict) or "class_type" not in node or "inputs" not in node:
                raise ValueError("Profile workflow must use ComfyUI API format")
        return graph

    @staticmethod
    def _input(body) -> tuple[dict, bytes | None]:
        if isinstance(body, MultipartBody):
            fields = {}
            for key in ALLOWED:
                value = body.field(key)
                if value is not None:
                    if key in ("seed", "width", "height", "steps", "n"):
                        value = int(value)
                    elif key == "cfg_scale":
                        value = float(value)
                    elif key == "async":
                        value = value.lower() == "true"
                    fields[key] = value
            return fields, body.raw("image") or body.raw("file")
        if not isinstance(body, dict):
            raise ValueError("image request must be JSON or multipart")
        return dict(body), None

    def _interrupt(self, model: str) -> None:
        try:
            with self.lock:
                port = self.running_port
            if port is None:
                return
            request = urllib.request.Request(
                f"http://127.0.0.1:{port}/api/jobs/current/cancel",
                data=b"{}", method="POST", headers={"Content-Type": "application/json"})
            urllib.request.urlopen(request, timeout=5).close()
        except Exception:
            pass

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
