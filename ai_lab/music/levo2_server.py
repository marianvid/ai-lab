#!/usr/bin/env python3
"""HTTP host for LeVo 2 music generation."""
from __future__ import annotations

import argparse
from pathlib import Path

from ai_lab.media.http_host import serve
from ai_lab.music.levo2_backend import Levo2Backend


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cantor", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--lm", required=True)
    parser.add_argument("--flow", required=True)
    parser.add_argument("--vae", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--cfg", type=float, required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    serve(Levo2Backend(args.cantor, args.model_path, args.lm, args.flow,
                       args.vae, args.output_root, args.steps, args.cfg),
          args.port)


if __name__ == "__main__":
    main()
