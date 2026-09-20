#!/usr/bin/env python3
"""Isolated HTTP host for configured ComfyUI video workflows."""
from __future__ import annotations

import argparse
import atexit
import signal
import sys
import tempfile
from pathlib import Path

from ai_lab.media.http_host import serve
from ai_lab.video.comfy_backend import ComfyVideoBackend


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comfyui", required=True)
    parser.add_argument("--workflow", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-path", action="append", required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    state = Path(tempfile.gettempdir()) / f"ai-lab-comfy-video-{args.port}"
    backend = ComfyVideoBackend(sys.executable, args.comfyui,
                                args.model_path, state, args.workflow,
                                args.model_name, args.port + 10000)
    atexit.register(backend.close)

    def stop(_signum, _frame):
        backend.close()
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    serve(backend, args.port, max_body_bytes=36 * 1024 * 1024,
          request_path="/v1/videos/generations")


if __name__ == "__main__":
    main()
