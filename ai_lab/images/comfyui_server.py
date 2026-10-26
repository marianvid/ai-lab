#!/usr/bin/env python3
"""Isolated bridge between AI-Lab and a private ComfyUI process."""

from __future__ import annotations

import argparse
import atexit
import base64
import contextlib
import copy
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from http.server import ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from typing import Any

from ai_lab.comfyui_templates import for_model
from ai_lab.engine_host.http_host import JsonReplies
from ai_lab.network import LOOPBACK, PAGE_INTERFACES

# How long one job may run, how long ComfyUI may take to start, how often
# the job's history is asked for.
DEFAULT_TIMEOUT_S = 1800.0
START_LIMIT_S = 180.0
HISTORY_POLL_S = 0.25
# The private ComfyUI listens this far above the bridge's own port.
PRIVATE_PORT_OFFSET = 10_000
VRAM_FLAGS = {"low": ("--lowvram",), "cpu": ("--cpu",)}


@dataclass(frozen=True)
class ComfyLaunch:
    """How to start one private ComfyUI: its program, models, state folder and port.

    `vram_mode` is "normal", "low" (`--lowvram`) or "cpu". `workflow` is the
    API workflow preset the bridge fills in; `ui_workflow` the one ComfyUI's
    own page opens with.
    """

    python: str
    comfyui: str
    model_roots: list[str]
    state: Path
    port: int
    timeout: float = DEFAULT_TIMEOUT_S
    vram_mode: str = "normal"
    workflow: Path | None = None
    ui_workflow: Path | None = None


def wait_until_ready(backend: Backend) -> None:
    """Return once the backend's ComfyUI answers; raise if it exits or never does."""
    deadline = time.monotonic() + START_LIMIT_S
    while time.monotonic() < deadline:
        if backend.process.poll() is not None:
            raise RuntimeError("ComfyUI exited during startup")
        try:
            backend.request("GET", "/system_stats")
            return
        except Exception:
            time.sleep(.5)
    raise TimeoutError("ComfyUI did not become ready")


class Backend:
    """A private ComfyUI process, started and stopped by this bridge, and its API.

    `wait_ready` is asked once the process is started (`wait_until_ready` by
    default); a test hands in one that does not wait.
    """

    def __init__(self, launch: ComfyLaunch,
                 wait_ready: Callable[[Backend], None] = wait_until_ready) -> None:
        self.base = f"http://{LOOPBACK}:{launch.port}"
        self.timeout = launch.timeout
        self.lock = Lock()
        launch.state.mkdir(parents=True, exist_ok=True)
        extra = launch.state / "extra_model_paths.yaml"
        extra.write_text(_model_paths(launch.model_roots))
        command = [launch.python, launch.comfyui, "--listen", PAGE_INTERFACES,
                   "--port", str(launch.port), "--extra-model-paths-config", str(extra),
                   "--output-directory", str(launch.state / "output"),
                   "--temp-directory", str(launch.state / "temp"),
                   *VRAM_FLAGS.get(launch.vram_mode, ())]
        self.process = subprocess.Popen(command, env=_environment(launch))
        wait_ready(self)

    def close(self) -> None:
        """Stop the private ComfyUI process owned by this bridge."""
        if self.process.poll() is not None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=5)

    def request(self, method: str, path: str, payload=None) -> dict | bytes:
        """One call to ComfyUI's API: JSON back as a dict, anything else as bytes."""
        data = None if payload is None else json.dumps(payload).encode()
        request = urllib.request.Request(self.base + path, data=data, method=method,
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=30) as response:
            body = response.read()
            return json.loads(body) if "json" in response.headers.get("Content-Type", "") else body

    def json(self, method: str, path: str, payload=None) -> dict:
        """One call to ComfyUI's API that must answer JSON."""
        answer = self.request(method, path, payload)
        if not isinstance(answer, dict):
            raise RuntimeError(f"ComfyUI answered {path} with something other than JSON")
        return answer

    def generate(self, payload: dict) -> dict:
        """Run the request's workflow (with its input image, if any) and return the images."""
        with self.lock:
            workflow = copy.deepcopy(payload["workflow"])
            image = payload.get("image_base64")
            if image:
                name = self._upload(base64.b64decode(image))
                replace_marker(workflow, "__AI_LAB_INPUT__", name)
            prompt_id = self.json("POST", "/prompt", {"prompt": workflow})["prompt_id"]
            deadline = time.monotonic() + self.timeout
            while time.monotonic() < deadline:
                history = self.json("GET", f"/history/{prompt_id}")
                if prompt_id in history:
                    return self._outputs(history[prompt_id])
                time.sleep(HISTORY_POLL_S)
            self.interrupt()
            raise TimeoutError("ComfyUI job timed out")

    def interrupt(self) -> None:
        """Stop the job ComfyUI is running; nothing if it cannot be reached."""
        with contextlib.suppress(Exception):
            self.request("POST", "/interrupt", {})

    def _upload(self, image: bytes) -> str:
        boundary = "ai-lab-comfyui"
        body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; "
                "filename=\"input.png\"\r\nContent-Type: image/png\r\n\r\n").encode()
        body += image + f"\r\n--{boundary}--\r\n".encode()
        request = urllib.request.Request(
            self.base + "/upload/image", data=body, method="POST",
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read())["name"]

    def _outputs(self, history: dict) -> dict:
        """Every image the finished job saved, read back from ComfyUI."""
        images = [self._image(item) for output in history.get("outputs", {}).values()
                  for item in output.get("images", [])]
        if not images:
            raise RuntimeError("ComfyUI completed without an image output")
        return {"data": images}

    def _image(self, item: dict) -> dict:
        """One saved image, base64-encoded, with its type."""
        wanted = {key: item[key] for key in ("filename", "subfolder", "type") if key in item}
        request = urllib.request.Request(f"{self.base}/view?{urllib.parse.urlencode(wanted)}")
        with urllib.request.urlopen(request, timeout=30) as response:
            return {"b64_json": base64.b64encode(response.read()).decode(),
                    "mime_type": response.headers.get_content_type()}


def replace_marker(value, wanted, replacement) -> None:
    """Replace every `wanted` value inside a workflow (nested dicts and lists), in place."""
    items = (value.items() if isinstance(value, dict)
             else enumerate(value) if isinstance(value, list) else ())
    for key, item in list(items):
        if item == wanted:
            value[key] = replacement
        else:
            replace_marker(item, wanted, replacement)


def _model_paths(model_roots: list[str]) -> str:
    """ComfyUI's extra model paths: every model root, then AI-Lab's preset node."""
    sections = []
    for index, model_root in enumerate(dict.fromkeys(model_roots)):
        root = str(Path(model_root).resolve())
        sections.append(
            f"ai_lab_{index}:\n  base_path: {json.dumps(root)}\n"
            "  checkpoints: .\n  diffusion_models: .\n"
            "  text_encoders: .\n  vae: .\n  loras: .\n"
            "  controlnet: .\n  clip_vision: .\n"
            "  latent_upscale_models: .\n")
    preset_nodes = Path(__file__).resolve().parents[1] / "comfyui_custom_nodes"
    sections.append(f"ai_lab_preset:\n  base_path: {json.dumps(str(preset_nodes))}\n"
                    "  custom_nodes: .\n")
    return "".join(sections)


def _environment(launch: ComfyLaunch) -> dict[str, str]:
    """This process's environment, with only this launch's presets named."""
    environment = os.environ.copy()
    environment.pop("AI_LAB_COMFYUI_PRESET", None)
    environment.pop("AI_LAB_COMFYUI_UI_PRESET", None)
    if launch.workflow is not None:
        environment["AI_LAB_COMFYUI_PRESET"] = str(launch.workflow.resolve())
    if launch.ui_workflow is not None:
        environment["AI_LAB_COMFYUI_UI_PRESET"] = str(launch.ui_workflow.resolve())
    return environment


class Handler(JsonReplies):
    """AI-Lab's image requests and the cancel button, answered through ComfyUI."""

    backend: Backend
    model_name = ""

    def _listing(self) -> dict | None:
        if self.path != "/v1/models":
            return None
        return {"object": "list", "data": [{"id": self.model_name, "object": "model",
                                            "owned_by": "comfyui"}]}

    def do_POST(self) -> None:  # noqa: N802 - the name http.server calls
        """Generate or edit an image, or cancel the job running now."""
        try:
            if self.path == "/api/jobs/current/cancel":
                self.backend.interrupt()
                self._json(200, {"ok": True})
                return
            if self.path not in ("/v1/images/generations", "/v1/images/edits"):
                self._not_found()
                return
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            self._json(200, self.backend.generate(payload))
        except Exception as error:
            self._json(400, {"error": {"message": str(error),
                                       "type": error.__class__.__name__}})

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - the base class's name
        """One line per request on standard output, where the manager reads the log."""
        print(format % args, flush=True)


def main() -> None:
    """Start the private ComfyUI for the model named on the command line, and answer."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--comfyui", required=True)
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--extra-model-root", action="append", default=[])
    parser.add_argument("--name", required=True)
    parser.add_argument("--host", default=LOOPBACK)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--workflow", type=Path)
    memory = parser.add_mutually_exclusive_group()
    memory.add_argument("--lowvram", action="store_true")
    memory.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    # Each bridge gets its own private ComfyUI port.  A fixed port left a
    # previous model's child process answering the next model after a switch.
    backend = Backend(ComfyLaunch(
        sys.executable, args.comfyui, [args.model_root, *args.extra_model_root],
        Path(tempfile.gettempdir()) / f"ai-lab-comfyui-{args.port}",
        args.port + PRIVATE_PORT_OFFSET,
        vram_mode="low" if args.lowvram else "cpu" if args.cpu else "normal",
        workflow=args.workflow, ui_workflow=for_model(args.name)))
    atexit.register(backend.close)

    def stop(_signum, _frame):
        backend.close()
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    handler = type("BoundHandler", (Handler,), {"backend": backend, "model_name": args.name})
    ThreadingHTTPServer((args.host, args.port), handler).serve_forever()


if __name__ == "__main__":
    main()
