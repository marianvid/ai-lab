#!/usr/bin/env python3
"""Isolated bridge between AI-Lab and a private ComfyUI process."""

from __future__ import annotations

import argparse
import base64
import copy
import json
import subprocess
import tempfile
import time
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock


class Backend:
    def __init__(self, python: str, comfyui: str, model_roots: list[str],
                 state: Path, timeout: float = 1800,
                 vram_mode: str = "normal") -> None:
        self.base = "http://127.0.0.1:8189"
        self.timeout = timeout
        self.lock = Lock()
        state.mkdir(parents=True, exist_ok=True)
        extra = state / "extra_model_paths.yaml"
        sections = []
        for index, model_root in enumerate(dict.fromkeys(model_roots)):
            root = str(Path(model_root).resolve())
            sections.append(
                f"ai_lab_{index}:\n  base_path: {json.dumps(root)}\n"
                "  checkpoints: .\n  diffusion_models: .\n"
                "  text_encoders: .\n  vae: .\n  loras: .\n"
                "  controlnet: .\n  clip_vision: .\n")
        extra.write_text("".join(sections))
        command = [python, comfyui, "--listen", "127.0.0.1", "--port", "8189",
                   "--extra-model-paths-config", str(extra),
                   "--output-directory", str(state / "output"),
                   "--temp-directory", str(state / "temp")]
        if vram_mode == "low":
            command.append("--lowvram")
        elif vram_mode == "cpu":
            command.append("--cpu")
        self.process = subprocess.Popen(command)
        self._wait_ready()

    def _wait_ready(self) -> None:
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError("ComfyUI exited during startup")
            try:
                self.request("GET", "/system_stats")
                return
            except Exception:
                time.sleep(.5)
        raise TimeoutError("ComfyUI did not become ready")

    def request(self, method: str, path: str, payload=None) -> dict | bytes:
        data = None if payload is None else json.dumps(payload).encode()
        request = urllib.request.Request(self.base + path, data=data, method=method,
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=30) as response:
            body = response.read()
            return json.loads(body) if "json" in response.headers.get("Content-Type", "") else body

    def generate(self, payload: dict) -> dict:
        with self.lock:
            workflow = copy.deepcopy(payload["workflow"])
            image = payload.get("image_base64")
            if image:
                name = self._upload(base64.b64decode(image))
                self._replace(workflow, "__AI_LAB_INPUT__", name)
            accepted = self.request("POST", "/prompt", {"prompt": workflow})
            prompt_id = accepted["prompt_id"]
            deadline = time.monotonic() + self.timeout
            while time.monotonic() < deadline:
                history = self.request("GET", f"/history/{prompt_id}")
                if prompt_id in history:
                    return self._outputs(history[prompt_id])
                time.sleep(.25)
            self.interrupt()
            raise TimeoutError("ComfyUI job timed out")

    def interrupt(self) -> None:
        try:
            self.request("POST", "/interrupt", {})
        except Exception:
            pass

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

    @staticmethod
    def _replace(value, wanted, replacement) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if item == wanted:
                    value[key] = replacement
                else:
                    Backend._replace(item, wanted, replacement)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                if item == wanted:
                    value[index] = replacement
                else:
                    Backend._replace(item, wanted, replacement)

    def _outputs(self, history: dict) -> dict:
        images = []
        for output in history.get("outputs", {}).values():
            for item in output.get("images", []):
                query = urllib.parse.urlencode({key: item[key] for key in
                                                ("filename", "subfolder", "type")
                                                if key in item})
                request = urllib.request.Request(self.base + "/view?" + query)
                with urllib.request.urlopen(request, timeout=30) as response:
                    images.append({"b64_json": base64.b64encode(response.read()).decode(),
                                   "mime_type": response.headers.get_content_type()})
        if not images:
            raise RuntimeError("ComfyUI completed without an image output")
        return {"data": images}


class Handler(BaseHTTPRequestHandler):
    backend: Backend
    model_name = ""

    def do_GET(self):
        if self.path == "/health":
            self._json(200, {"status": "ok"})
        elif self.path == "/v1/models":
            self._json(200, {"object": "list", "data": [{"id": self.model_name,
                                                            "object": "model",
                                                            "owned_by": "comfyui"}]})
        else:
            self._json(404, {"error": {"message": "not found"}})

    def do_POST(self):
        try:
            if self.path == "/api/jobs/current/cancel":
                self.backend.interrupt()
                self._json(200, {"ok": True})
                return
            if self.path not in ("/v1/images/generations", "/v1/images/edits"):
                self._json(404, {"error": {"message": "not found"}})
                return
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            self._json(200, self.backend.generate(payload))
        except Exception as error:
            self._json(400, {"error": {"message": str(error),
                                        "type": error.__class__.__name__}})

    def _json(self, status, payload):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        print(format % args, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comfyui", required=True)
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--extra-model-root", action="append", default=[])
    parser.add_argument("--name", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    memory = parser.add_mutually_exclusive_group()
    memory.add_argument("--lowvram", action="store_true")
    memory.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    state = Path(tempfile.gettempdir()) / f"ai-lab-comfyui-{args.port}"
    vram_mode = "low" if args.lowvram else "cpu" if args.cpu else "normal"
    Handler.backend = Backend(__import__("sys").executable, args.comfyui,
                              [args.model_root, *args.extra_model_root], state,
                              vram_mode=vram_mode)
    Handler.model_name = args.name
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
