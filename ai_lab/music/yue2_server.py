#!/usr/bin/env python3
"""Isolated HTTP host for YuE2 music generation."""
from __future__ import annotations

import argparse
from pathlib import Path

from ai_lab.music.http_host import serve
from ai_lab.music.yue2_backend import Yue2Backend


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--vae-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cot", choices=("full", "melody"), required=True)
    parser.add_argument("--memory-budget-gib", type=float, required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    backend = Yue2Backend(args.model_path, args.vae_path, args.output_root,
                          args.cot, args.memory_budget_gib)
    serve(backend, args.port)


if __name__ == "__main__":
    main()
