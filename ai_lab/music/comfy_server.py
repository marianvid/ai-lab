#!/usr/bin/env python3
"""Isolated MiniMax-style ComfyUI music workflow host."""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

from ai_lab.media.http_host import serve
from ai_lab.music.comfy_backend import ComfyMusicBackend, ComfyRuntime
from ai_lab.music.hosting import PORT, parse, stop_on_exit

# The private ComfyUI listens this far above the music host's own port.
COMFY_PORT_OFFSET = 10000


def main() -> None:
    """Start the private ComfyUI with the workflow, then answer music requests."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", action="append", required=True)
    args = parse(("--comfyui", str), ("--workflow", Path), ("--model-name", str),
                 ("--ffmpeg", str), PORT, parser=parser)
    state = Path(tempfile.gettempdir()) / f"ai-lab-comfy-music-{args.port}"
    runtime = ComfyRuntime(sys.executable, args.comfyui, args.model_path, state,
                           args.port + COMFY_PORT_OFFSET)
    backend = ComfyMusicBackend(runtime, args.workflow, args.model_name, args.ffmpeg)
    stop_on_exit(backend.close)
    serve(backend, args.port)


if __name__ == "__main__":
    main()
