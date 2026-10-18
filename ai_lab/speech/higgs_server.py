#!/usr/bin/env python3
"""HTTP host that manages one Higgs worker and exposes AI-Lab speech JSON.

On Linux, Higgs runs inside an SGLang-Omni worker process (`HiggsBackend`
starts and stops it). Beside it this starts the playground page that ships
with SGLang-Omni, waits until the page answers, then answers AI-Lab's speech
contract on `--port` (`engine_host.http_host`). Stopping the host stops both.
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from http import HTTPStatus
from http.server import ThreadingHTTPServer
from pathlib import Path
from threading import Thread

from ai_lab.engine_host.http_host import GenerationHost
from ai_lab.network import ALL_INTERFACES, LOOPBACK
from ai_lab.speech.contract import MAX_REQUEST_BYTES, PATH
from ai_lab.speech.higgs_backend import HiggsBackend

# How long the playground page may take to come up, and how often to look.
PLAYGROUND_START_S = 30.0
PLAYGROUND_POLL_S = 0.5


def main() -> None:
    """Start the worker and the playground, then answer speech requests until stopped."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-binary", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--worker-port", type=int, required=True)
    parser.add_argument("--mem-fraction-static", type=float, required=True)
    parser.add_argument("--max-parallel", type=int, default=1)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--ui-port", type=int, required=True)
    args = parser.parse_args()
    backend = HiggsBackend(args.worker_binary, args.model_path,
                           args.worker_port, args.mem_fraction_static,
                           args.max_parallel)
    try:
        playground = _start_playground(args.worker_port, args.ui_port)
    except Exception:
        backend.close()
        raise
    try:
        _wait_for_playground(playground, args.ui_port)
        server = ThreadingHTTPServer((ALL_INTERFACES, args.port), _host_for(backend))
    except Exception:
        _stop(playground)
        backend.close()
        raise
    signal.signal(signal.SIGTERM, lambda *_: Thread(
        target=server.shutdown, daemon=True).start())
    try:
        server.serve_forever()
    finally:
        _stop(playground)
        backend.close()
        server.server_close()


def _host_for(backend: HiggsBackend) -> type[GenerationHost]:
    """The speech handler, bound to this worker."""
    return type("HiggsHost", (GenerationHost,), {
        "backend": backend, "max_body_bytes": MAX_REQUEST_BYTES,
        "request_path": PATH, "noun": "speech"})


def _start_playground(worker_port: int, ui_port: int) -> subprocess.Popen:
    """SGLang-Omni's own playground page, talking to the worker."""
    playground_root = Path(__file__).resolve().parents[1] / "vendor" / "sglang_omni"
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (
        str(playground_root), env.get("PYTHONPATH", ""))))
    return subprocess.Popen([
        sys.executable, "-m", "playground.higgs.app", "--api-base",
        f"http://{LOOPBACK}:{worker_port}", "--port", str(ui_port)], env=env)


def _wait_for_playground(playground: subprocess.Popen, ui_port: int) -> None:
    """Until the page answers its health check; an error if it exits or never does."""
    deadline = time.monotonic() + PLAYGROUND_START_S
    while time.monotonic() < deadline:
        if playground.poll() is not None:
            raise RuntimeError("Higgs playground exited during startup")
        try:
            with urllib.request.urlopen(
                    f"http://{LOOPBACK}:{ui_port}/healthz", timeout=2) as response:
                if response.status == HTTPStatus.OK:
                    return
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(PLAYGROUND_POLL_S)
    raise TimeoutError("Higgs playground did not become ready")


def _stop(playground: subprocess.Popen) -> None:
    """Stop the playground: politely, then for good."""
    if playground.poll() is not None:
        return
    playground.terminate()
    try:
        playground.wait(timeout=10)
    except subprocess.TimeoutExpired:
        playground.kill()
        playground.wait(timeout=5)


if __name__ == "__main__":
    main()
