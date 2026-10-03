"""Proxy SGLang-Omni's binary speech response to AI-Lab's JSON contract."""
from __future__ import annotations

import base64
import json
import os
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from threading import BoundedSemaphore

from .contract import ReferenceFile, validate_payload


def wait_for_free_port(port: int, timeout_s: float = 120) -> None:
    """Wait until `port` on 127.0.0.1 can be bound, the way the worker checks.

    No SO_REUSEADDR, on purpose: the worker binds without it, so a port in
    TIME_WAIT counts as taken for it, and must count as taken here too.
    """
    deadline = time.monotonic() + timeout_s
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            try:
                probe.bind(("127.0.0.1", port))
                return
            except OSError:
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        f"Higgs worker port {port} is still in use") from None
        time.sleep(1)


class HiggsBackend:
    def __init__(self, worker_binary: Path, model_path: Path,
                 worker_port: int, mem_fraction_static: float,
                 max_parallel: int = 1) -> None:
        if not worker_binary.is_file() or not model_path.is_dir():
            raise ValueError("Higgs worker or checkpoint is absent")
        if (not 1 <= worker_port <= 65535 or not 0 < mem_fraction_static < 1
                or not 1 <= max_parallel <= 64):
            raise ValueError("Higgs worker settings are invalid")
        self.model_name = model_path.name
        self.model_path = model_path
        self.worker_port = worker_port
        # The worker batches concurrent requests; this caps how many it is
        # handed at once, as the gateway does, for callers that reach this
        # host without going through the gateway.
        self.slots = BoundedSemaphore(max_parallel)
        # A worker stopped moments ago leaves its port held by the kernel for
        # about a minute. The worker then quietly picks another port, and this
        # host would wait ten minutes on the old one. So wait for the port
        # first, and tell the worker to stop rather than move if it is taken.
        wait_for_free_port(worker_port)
        self.process = subprocess.Popen([
            str(worker_binary), "serve", "--model-path", str(model_path),
            "--host", "127.0.0.1", "--port", str(worker_port),
            "--mem-fraction-static", str(mem_fraction_static),
            "--log-level", "info"],
            env={**os.environ, "SGLANG_OMNI_STRICT_PORT": "1"})
        try:
            self._wait_ready()
        except Exception:
            self.close()
            raise

    def _wait_ready(self) -> None:
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError("Higgs worker exited during startup")
            try:
                with urllib.request.urlopen(
                        f"http://127.0.0.1:{self.worker_port}/health",
                        timeout=2) as response:
                    if response.status == 200:
                        return
            except (OSError, urllib.error.URLError):
                pass
            time.sleep(2)
        raise TimeoutError("Higgs worker did not become ready")

    def generate(self, body: dict) -> dict:
        request = validate_payload(body, seed=True, reference=True)
        if body.get("model") not in (None, self.model_name):
            raise ValueError(f"this engine serves {self.model_name}")
        if request["instruction"]:
            raise ValueError("Higgs accepts style controls inline in the text")
        with ReferenceFile(request["reference_audio"]) as reference:
            fields = {"model": str(self.model_path), "input": request["text"],
                      "voice": request["speaker"] or "default",
                      "response_format": "wav"}
            if request["seed"] is not None:
                fields["seed"] = request["seed"]
            if reference:
                # The worker runs on this machine and reads the clip by path;
                # the transcript is what makes the cloning faithful.
                fields["references"] = [{"audio_path": reference,
                                         "text": request["reference_text"] or None}]
            call = urllib.request.Request(
                f"http://127.0.0.1:{self.worker_port}/v1/audio/speech",
                data=json.dumps(fields).encode(),
                headers={"Content-Type": "application/json"})
            with self.slots, urllib.request.urlopen(call, timeout=600) as response:
                audio = response.read()
        if not audio.startswith(b"RIFF"):
            raise RuntimeError("Higgs did not return WAV audio")
        return {"model": self.model_name, "mode": "higgs",
                "data": [{"mime_type": "audio/wav",
                          "b64_wav": base64.b64encode(audio).decode()}]}

    def close(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
