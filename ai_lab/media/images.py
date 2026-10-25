"""Image calls by named profile: an ordinary call, answered when the image is made.

No job list, job id or polling (the approved list, A10): the call waits in the
card's queue like any other (A3), the profile's ComfyUI graph is filled in
(`image_workflows`), sent to the engine, and its images come back in the answer.
"""

from __future__ import annotations

import base64
import json
import time
import urllib.request
from pathlib import Path

from ..multipart import MultipartBody
from ..network import LOOPBACK
from ..types import Task
from ..uploads import validate_image
from .image_workflows import ALLOWED, read_request, render

EDIT_PATH = "/v1/images/edits"
GENERATION_PATH = "/v1/images/generations"


class ImageCalls:
    """The named profiles, and one image call through the gateway."""

    def __init__(self, gateway, settings: dict) -> None:
        self.gateway = gateway
        self.settings = settings or {}

    def profiles(self) -> list[dict]:
        """The named profiles a client may ask for."""
        return [{"id": key, "task": value.get("task", "generation"),
                 "model": value.get("model", ""),
                 "description": value.get("description", "")}
                for key, value in sorted(self.settings.get("profiles", {}).items())]

    def submit(self, body: dict | MultipartBody, task: Task) -> dict:
        """Make the images the request asks for, and answer with them."""
        payload, image = read_request(body)
        profile_id, profile = self._profile(payload, task, image)
        model = str(profile.get("model") or payload.get("model") or "")
        if not model:
            raise ValueError(f"Profile {profile_id} has no model")
        work = {"workflow": render(profile, payload,
                                   Path(self.settings.get("workflow_root", "."))),
                "image_base64": base64.b64encode(image).decode() if image else None}
        result = self._call(model, task, work)
        if not result.get("data"):
            raise RuntimeError("image engine returned no images")
        return {"created": int(time.time()), "profile": profile_id, "data": result["data"]}

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
            validate_image(image, max_bytes=self.gateway.max_upload_bytes,
                           max_pixels=self.gateway.max_upload_pixels,
                           max_dimension=self.gateway.max_upload_dimension)
        return profile_id, profile

    def _call(self, model: str, task: Task, work: dict) -> dict:
        """The engine's answer, holding a place on the card for the whole of it."""
        path = EDIT_PATH if task is Task.IMAGE_EDIT else GENERATION_PATH
        with self.gateway.acquire(model, shape=path) as lease:
            request = urllib.request.Request(
                f"http://{LOOPBACK}:{lease.port}{path}", data=json.dumps(work).encode(),
                method="POST", headers={"Content-Type": "application/json"})
            first, _ = self.gateway.timeouts_for(task)
            with urllib.request.urlopen(request, timeout=first) as response:
                return json.loads(response.read())
