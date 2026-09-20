#!/usr/bin/env python3
"""Isolated HTTP host for MuLaCover source-audio remixes."""
from __future__ import annotations

import argparse
from pathlib import Path

from ai_lab.media.http_host import serve
from ai_lab.music.mulacover_backend import MulaCoverBackend


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--checkpoint-subdir", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--topk", type=int, required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--cfg-scale", type=float, required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    backend = MulaCoverBackend(
        args.bundle_root, args.checkpoint_path, args.checkpoint_subdir,
        args.output_root, args.topk, args.temperature, args.cfg_scale)
    serve(backend, args.port, max_body_bytes=36 * 1024 * 1024)


if __name__ == "__main__":
    main()
