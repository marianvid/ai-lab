"""Proxy SGLang-Omni's binary speech response to AI-Lab's JSON contract."""
from __future__ import annotations

import base64
import json
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from threading import Lock

from .contract import validate_payload


class HiggsBackend:
    def __init__(self, worker_binary: Path, model_path: Path,
                 worker_port: int, mem_fraction_static: float) -> None:
        if not worker_binary.is_file() or not model_path.is_dir():
            raise ValueError("Higgs worker or checkpoint is absent")
        if not 1 <= worker_port <= 65535 or not 0 < mem_fraction_static < 1:
            raise ValueError("Higgs worker settings are invalid")
        self.model_name = model_path.name
        self.model_path = model_path
        self.worker_port = worker_port
        self.lock = Lock()
        self.process = subprocess.Popen([
            str(worker_binary), "serve", "--model-path", str(model_path),
            "--host", "127.0.0.1", "--port", str(worker_port),
            "--mem-fraction-static", str(mem_fraction_static),
            "--log-level", "info"])
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
        request = validate_payload(body)
        if body.get("model") not in (None, self.model_name):
            raise ValueError(f"this engine serves {self.model_name}")
        if request["instruction"]:
            raise ValueError("Higgs accepts style controls inline in the text")
        payload = json.dumps({"model": str(self.model_path),
                              "input": request["text"],
                              "voice": request["speaker"] or "default",
                              "response_format": "wav"}).encode()
        call = urllib.request.Request(
            f"http://127.0.0.1:{self.worker_port}/v1/audio/speech",
            data=payload, headers={"Content-Type": "application/json"})
        with self.lock, urllib.request.urlopen(call, timeout=600) as response:
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
