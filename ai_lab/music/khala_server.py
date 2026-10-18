#!/usr/bin/env python3
"""Isolated HTTP host for the Khala music generator.

Besides AI-Lab's music contract, this starts Khala's own app for a person: its
Mac worker (which keeps the model loaded), its dispatcher, and its React page.
They are started in that order, each waited for, and stopped with this host.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from ai_lab.media.http_host import serve
from ai_lab.music.hosting import (
    PORT,
    UI_PORT,
    Readiness,
    parse,
    stop_on_exit,
    wait_until_ready,
)
from ai_lab.music.khala_backend import KhalaBackend
from ai_lab.network import ALL_INTERFACES, LOOPBACK

# Where Khala's own worker and dispatcher listen (their defaults).
WORKER_PORT = 8001
API_PORT = 8889
# How long each may take to come up: the worker loads the model.
WORKER_START_S = 360
API_START_S = 30
PAGE_START_S = 30
# How long the companions get to stop before they are killed.
STOP_GRACE_S = 8


def main() -> None:
    """Start Khala's own app, then answer music requests."""
    args = parse(("--generator-script", Path), ("--model-path", Path),
                 ("--output-root", Path), ("--default-bucket", int),
                 ("--maximum-bucket", int), PORT, UI_PORT)
    backend = KhalaBackend(args.generator_script, args.model_path, args.output_root,
                           (args.default_bucket, args.maximum_bucket))
    launch_official_ui(args.generator_script.parent.parent, args.model_path, args.ui_port)
    serve(backend, args.port)


def launch_official_ui(project_root: Path, model_path: Path, ui_port: int) -> None:
    """Run Khala's complete Mac worker, dispatcher and React frontend."""
    frontend = project_root / "frontend"
    worker = project_root / "backend" / "backend_worker.py"
    api = project_root / "backend" / "backend_api.py"
    if not all(path.exists() for path in (worker, api, frontend / "package.json",
                                           frontend / "node_modules")):
        raise RuntimeError("Khala's official frontend runtime is incomplete")
    environment = {**os.environ, "KHALA_BACKEND": "vanilla", "KHALA_DEVICE": "mps",
                   "KHALA_VANILLA_WEIGHTS": str(model_path),
                   "KHALA_TRACKS_PER_JOB": "1", "PYTHONPATH": str(project_root)}
    children: list[subprocess.Popen] = []
    stop_on_exit(lambda: _stop_all(children))

    def start(command: list[str], cwd: Path, readiness: Readiness) -> None:
        children.append(subprocess.Popen(command, cwd=cwd, env=environment))
        wait_until_ready(readiness, "Khala companion")
    start([sys.executable, str(worker), "--worker-port", str(WORKER_PORT),
           "--runtime-mode", "keep_loaded"], project_root,
          Readiness(f"http://{LOOPBACK}:{WORKER_PORT}/health", WORKER_START_S,
                    _says("status", "idle")))
    start([sys.executable, str(api), "--port", str(API_PORT), "--num-workers", "1",
           "--worker-base-port", str(WORKER_PORT)], project_root,
          Readiness(f"http://{LOOPBACK}:{API_PORT}/status", API_START_S,
                    _says("total_gpus", 1)))
    start(["npm", "run", "dev", "--", "--host", ALL_INTERFACES, "--port", str(ui_port)],
          frontend, Readiness(f"http://{LOOPBACK}:{ui_port}/", PAGE_START_S))


def _says(key: str, expected):
    """A readiness test: the JSON answer has `key` set to `expected`."""
    return lambda response: json.load(response).get(key) == expected


def _stop_all(children: list[subprocess.Popen]) -> None:
    """Stop the companions, newest first, within one shared grace period."""
    for child in reversed(children):
        if child.poll() is None:
            child.terminate()
    deadline = time.monotonic() + STOP_GRACE_S
    for child in reversed(children):
        if child.poll() is None:
            try:
                child.wait(timeout=max(0.1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                child.kill()


if __name__ == "__main__":
    main()
