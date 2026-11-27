#!/usr/bin/env python3
"""Isolated MiniMax-style ComfyUI music workflow host."""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

from ai_lab.engine_host.http_host import serve
from ai_lab.music.backends.comfy_backend import ComfyMusicBackend, ComfyRuntime
from ai_lab.music.hosting import PORT, UI_PORT, parse, stop_on_exit


def main() -> None:
    """Start the private ComfyUI with the workflow, then answer music requests."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", action="append", required=True)
    args = parse(("--comfyui", str), ("--workflow", Path), ("--model-name", str),
                 ("--ffmpeg", str), PORT, UI_PORT, parser=parser)
    state = Path(tempfile.gettempdir()) / f"ai-lab-comfy-music-{args.port}"
    runtime = ComfyRuntime(sys.executable, args.comfyui, args.model_path, state,
                           args.ui_port)
    backend = ComfyMusicBackend(runtime, args.workflow, args.model_name, args.ffmpeg)
    stop_on_exit(backend.close)
    serve(backend, args.port)


if __name__ == "__main__":
    main()
