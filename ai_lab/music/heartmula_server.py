#!/usr/bin/env python3
"""Isolated HTTP host for lyric-conditioned HeartMuLa generation."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from ai_lab.music.heartmula_backend import HeartMulaBackend
from ai_lab.media.http_host import serve


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--checkpoint-subdir", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--topk", type=int, required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--cfg-scale", type=float, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--ui-port", type=int, required=True)
    parser.add_argument("--heartmuse-root", type=Path, required=True)
    args = parser.parse_args()
    backend = HeartMulaBackend(
        args.bundle_root, args.checkpoint_path, args.checkpoint_subdir,
        args.output_root, args.version, args.topk,
        args.temperature, args.cfg_scale)
    launch_heartmuse(backend, args)
    serve(backend, args.port)


def launch_heartmuse(backend: HeartMulaBackend, args) -> None:
    """Run HeartMuse's complete UI around AI-Lab's resident pipeline."""
    if not (args.heartmuse_root / "app.py").is_file():
        raise RuntimeError("HeartMuse is not installed")
    os.environ.update({
        "CKPT_DIR": str(args.bundle_root),
        "OUTPUT_DIR": str(args.output_root / "heartmuse"),
        "MODEL_VARIANT": "base", "LAZY_LOAD": "true",
        "HF_HUB_OFFLINE": "1", "SERVER_HOST": "0.0.0.0",
        "SERVER_PORT": str(args.ui_port), "STYLE_TRANSFER": "true",
        "TRANSCRIPTION": "true",
    })
    sys.path.insert(0, str(args.heartmuse_root))
    import config as heart_config
    base = heart_config.MODEL_VARIANTS["base"]
    heart_config.MODEL_VARIANTS.clear()
    heart_config.MODEL_VARIANTS["base"] = base
    heart_config.MODEL_VARIANT_LABELS.clear()
    heart_config.MODEL_VARIANT_LABELS["base"] = "HeartMuLa 3B"
    heart_config.DEFAULT_MODEL_VARIANT = "base"
    import generator
    generator._pipeline = backend.pipeline
    generator._active_lazy_load = True
    generator._active_variant = "base"
    original_generate = generator.generate_music
    def serialized_generate(*values, **options):
        with backend.lock:
            return original_generate(*values, **options)
    generator.generate_music = serialized_generate
    generator.unload_pipeline = lambda: None
    import app as heartmuse
    heartmuse.app.queue(default_concurrency_limit=1).launch(
        server_name="0.0.0.0", server_port=args.ui_port,
        allowed_paths=[heart_config.OUTPUT_DIR], js=heartmuse.PLAYLIST_JS,
        share=False, inbrowser=False, prevent_thread_lock=True,
        show_error=True)


if __name__ == "__main__":
    main()
