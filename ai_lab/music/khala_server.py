#!/usr/bin/env python3
"""Isolated HTTP host for the Khala music generator."""
from __future__ import annotations

import argparse
import atexit
import os
import signal
import subprocess
import time
import urllib.request
from pathlib import Path

from ai_lab.music.khala_backend import KhalaBackend
from ai_lab.media.http_host import serve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator-script", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--default-bucket", type=int, required=True)
    parser.add_argument("--maximum-bucket", type=int, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--ui-port", type=int, required=True)
    args = parser.parse_args()
    backend = KhalaBackend(
        args.generator_script, args.model_path, args.output_root,
        args.default_bucket, args.maximum_bucket)
    launch_official_ui(args.generator_script.parent.parent, args.model_path,
                       args.ui_port)
    serve(backend, args.port)


def launch_official_ui(project_root: Path, model_path: Path,
                       ui_port: int) -> None:
    """Run Khala's complete Mac worker, dispatcher and React frontend."""
    worker_port = 8001
    api_port = 8889
    python = Path(os.sys.executable)
    frontend = project_root / "frontend"
    required = (project_root / "backend" / "backend_worker.py",
                project_root / "backend" / "backend_api.py",
                frontend / "package.json", frontend / "node_modules")
    if not all(path.exists() for path in required):
        raise RuntimeError("Khala's official frontend runtime is incomplete")
    environment = os.environ.copy()
    environment.update({
        "KHALA_BACKEND": "vanilla", "KHALA_DEVICE": "mps",
        "KHALA_VANILLA_WEIGHTS": str(model_path),
        "KHALA_TRACKS_PER_JOB": "1", "PYTHONPATH": str(project_root),
    })
    children: list[subprocess.Popen] = []

    def start(command: list[str], cwd: Path) -> subprocess.Popen:
        child = subprocess.Popen(command, cwd=cwd, env=environment)
        children.append(child)
        return child

    def stop_children() -> None:
        for child in reversed(children):
            if child.poll() is None:
                child.terminate()
        deadline = time.monotonic() + 8
        for child in reversed(children):
            if child.poll() is None:
                try:
                    child.wait(timeout=max(0.1, deadline - time.monotonic()))
                except subprocess.TimeoutExpired:
                    child.kill()

    atexit.register(stop_children)
    def terminate(_signum, _frame) -> None:
        stop_children()
        raise SystemExit(0)
    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    start([str(python), str(project_root / "backend" / "backend_worker.py"),
           "--worker-port", str(worker_port), "--runtime-mode", "keep_loaded"],
          project_root)
    _wait_json(f"http://127.0.0.1:{worker_port}/health", "status", "idle", 360)
    start([str(python), str(project_root / "backend" / "backend_api.py"),
           "--port", str(api_port), "--num-workers", "1",
           "--worker-base-port", str(worker_port)], project_root)
    _wait_json(f"http://127.0.0.1:{api_port}/status", "total_gpus", 1, 30)
    start(["npm", "run", "dev", "--", "--host", "0.0.0.0",
           "--port", str(ui_port)], frontend)
    _wait_json(f"http://127.0.0.1:{ui_port}/", None, None, 30)


def _wait_json(url: str, key: str | None, expected, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if key is None:
                    return
                import json
                body = json.load(response)
                if body.get(key) == expected:
                    return
        except Exception as error:
            last_error = error
        time.sleep(1)
    raise RuntimeError(f"Khala companion did not become ready at {url}: {last_error}")


if __name__ == "__main__":
    main()
