#!/usr/bin/env python3
"""Isolated HTTP host for the Khala music generator."""
from __future__ import annotations

import argparse
from pathlib import Path

from ai_lab.music.khala_backend import KhalaBackend
from ai_lab.music.http_host import serve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator-script", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--default-bucket", type=int, required=True)
    parser.add_argument("--maximum-bucket", type=int, required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    backend = KhalaBackend(
        args.generator_script, args.model_path, args.output_root,
        args.default_bucket, args.maximum_bucket)
    serve(backend, args.port)


if __name__ == "__main__":
    main()
