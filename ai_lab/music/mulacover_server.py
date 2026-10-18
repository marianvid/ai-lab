#!/usr/bin/env python3
"""Isolated HTTP host for MuLaCover source-audio remixes."""
from __future__ import annotations

from pathlib import Path

from ai_lab.media.http_host import serve
from ai_lab.music.bundles import Checkpoint, Sampling
from ai_lab.music.hosting import CHECKPOINT_OPTIONS, PORT, SAMPLING_OPTIONS, parse
from ai_lab.music.mulacover_backend import MulaCoverBackend

# Room for a base64 reference recording of up to 25 MiB.
MAX_BODY_BYTES = 36 * 1024 * 1024


def main() -> None:
    """Load MuLaCover, then answer cover requests."""
    args = parse(*CHECKPOINT_OPTIONS, ("--output-root", Path), *SAMPLING_OPTIONS, PORT)
    backend = MulaCoverBackend(
        Checkpoint(args.bundle_root, args.checkpoint_path, args.checkpoint_subdir),
        args.output_root, Sampling(args.topk, args.temperature, args.cfg_scale))
    serve(backend, args.port, max_body_bytes=MAX_BODY_BYTES)


if __name__ == "__main__":
    main()
