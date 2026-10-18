#!/usr/bin/env python3
"""Isolated HTTP host for lyric-conditioned HeartMuLa generation."""
from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path
from typing import Any

from ai_lab.media.http_host import serve
from ai_lab.music.bundles import Checkpoint, Sampling
from ai_lab.music.heartmula_backend import HeartMulaBackend
from ai_lab.music.hosting import CHECKPOINT_OPTIONS, PORT, SAMPLING_OPTIONS, UI_PORT, parse
from ai_lab.network import ALL_INTERFACES


def main() -> None:
    """Load HeartMuLa and the HeartMuse page around it, then answer music requests."""
    args = parse(*CHECKPOINT_OPTIONS, ("--output-root", Path), ("--version", str),
                 *SAMPLING_OPTIONS, PORT, UI_PORT, ("--heartmuse-root", Path))
    backend = HeartMulaBackend(
        Checkpoint(args.bundle_root, args.checkpoint_path, args.checkpoint_subdir),
        args.output_root, args.version, Sampling(args.topk, args.temperature, args.cfg_scale))
    launch_heartmuse(backend, args)
    serve(backend, args.port)


def launch_heartmuse(backend: HeartMulaBackend, args: argparse.Namespace) -> None:
    """Run HeartMuse's complete UI around AI-Lab's resident pipeline."""
    if not (args.heartmuse_root / "app.py").is_file():
        raise RuntimeError("HeartMuse is not installed")
    os.environ.update({
        "CKPT_DIR": str(args.bundle_root),
        "OUTPUT_DIR": str(args.output_root / "heartmuse"),
        "MODEL_VARIANT": "base", "LAZY_LOAD": "true",
        "HF_HUB_OFFLINE": "1", "SERVER_HOST": ALL_INTERFACES,
        "SERVER_PORT": str(args.ui_port), "STYLE_TRANSFER": "true",
        "TRANSCRIPTION": "true",
    })
    # HeartMuse's own modules, found through its folder at run time.
    sys.path.insert(0, str(args.heartmuse_root))
    heart_config: Any = importlib.import_module("config")
    base = heart_config.MODEL_VARIANTS["base"]
    heart_config.MODEL_VARIANTS.clear()
    heart_config.MODEL_VARIANTS["base"] = base
    heart_config.MODEL_VARIANT_LABELS.clear()
    heart_config.MODEL_VARIANT_LABELS["base"] = "HeartMuLa 3B"
    heart_config.DEFAULT_MODEL_VARIANT = "base"
    generator: Any = importlib.import_module("generator")
    generator._pipeline = backend.pipeline
    generator._active_lazy_load = True
    generator._active_variant = "base"
    original_generate = generator.generate_music
    def serialized_generate(*values, **options):
        with backend.lock:
            return original_generate(*values, **options)
    generator.generate_music = serialized_generate
    generator.unload_pipeline = lambda: None
    heartmuse: Any = importlib.import_module("app")
    heartmuse.app.queue(default_concurrency_limit=1).launch(
        server_name=ALL_INTERFACES, server_port=args.ui_port,
        allowed_paths=[heart_config.OUTPUT_DIR], js=heartmuse.PLAYLIST_JS,
        share=False, inbrowser=False, prevent_thread_lock=True,
        show_error=True)


if __name__ == "__main__":
    main()
