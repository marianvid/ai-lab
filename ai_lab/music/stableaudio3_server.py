#!/usr/bin/env python3
"""HTTP host for Stable Audio 3 instrumental music."""
from __future__ import annotations

import argparse
from pathlib import Path

from ai_lab.media.http_host import serve
from ai_lab.music.stableaudio3_backend import StableAudio3Backend


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--cfg-scale", type=float, required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    serve(StableAudio3Backend(args.model_path, args.steps, args.cfg_scale),
          args.port)


if __name__ == "__main__":
    main()
