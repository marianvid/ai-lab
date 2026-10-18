#!/usr/bin/env python3
"""HTTP host for LeVo 2 music generation."""
from __future__ import annotations

from pathlib import Path

from ai_lab.media.http_host import serve
from ai_lab.music.hosting import PORT, parse
from ai_lab.music.levo2_backend import Levo2Backend, Levo2Files


def main() -> None:
    """Check the runtime and model files, then answer music requests."""
    args = parse(("--cantor", Path), ("--model-path", Path), ("--lm", str),
                 ("--flow", str), ("--vae", str), ("--output-root", Path),
                 ("--steps", int), ("--cfg", float), PORT)
    files = Levo2Files(args.cantor, args.model_path, args.lm, args.flow, args.vae)
    serve(Levo2Backend(files, args.output_root, args.steps, args.cfg), args.port)


if __name__ == "__main__":
    main()
