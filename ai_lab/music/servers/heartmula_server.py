#!/usr/bin/env python3
"""Isolated HTTP host for lyric-conditioned HeartMuLa generation."""
from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path
from typing import Any

from ai_lab.engine_host.http_host import serve
from ai_lab.music.bundles import Checkpoint, Sampling
from ai_lab.music.backends.heartmula_backend import HeartMulaBackend
from ai_lab.music.hosting import CHECKPOINT_OPTIONS, PORT, SAMPLING_OPTIONS, UI_PORT, parse
from ai_lab.network import PAGE_INTERFACES

# HeartMuse's own names this host sets or reads to run its page around AI-Lab's
# pipeline, and the checkout they were read from. A newer HeartMuse may rename
# them without a word, so they are checked before anything is set
# (`check_heartmuse`), not found missing when the page is first used.
HEARTMUSE_WRITTEN_AGAINST = "strnad/HeartMuse a28600c5d6ba (2026-03-09)"
HEARTMUSE_NAMES = {
    "config": ("MODEL_VARIANTS", "MODEL_VARIANT_LABELS", "DEFAULT_MODEL_VARIANT", "OUTPUT_DIR"),
    "generator": ("_pipeline", "_active_lazy_load", "_active_variant", "generate_music",
                  "unload_pipeline"),
    "app": ("app", "PLAYLIST_JS"),
}


def missing_heartmuse_names(**modules: Any) -> list[str]:
    """The names used here that these HeartMuse modules (`config=`, `generator=`, `app=`) lack."""
    return [f"{module}.{name}" for module, found in modules.items()
            for name in HEARTMUSE_NAMES[module] if not hasattr(found, name)]


def check_heartmuse(**modules: Any) -> None:
    """Refuse to start when one of HeartMuse's modules no longer has a name used here."""
    missing = missing_heartmuse_names(**modules)
    if missing:
        raise RuntimeError(f"HeartMuse lacks {', '.join(missing)}; AI-Lab's page wiring was "
                           f"written against {HEARTMUSE_WRITTEN_AGAINST}")


def main() -> None:
    """Load HeartMuLa and the HeartMuse page around it, then answer music requests."""
    args = parse(*CHECKPOINT_OPTIONS, ("--output-root", Path), ("--version", str),
                 *SAMPLING_OPTIONS, PORT, UI_PORT, ("--heartmuse-root", Path))
    backend = HeartMulaBackend(
        Checkpoint(args.bundle_root, args.checkpoint_path, args.checkpoint_subdir),
        args.output_root, args.version, Sampling(args.topk, args.temperature, args.cfg_scale))
    launch_heartmuse(backend, args)
    serve(backend, args.port)


def _base_only(heart_config: Any) -> Any:
    """HeartMuse's settings narrowed to the one checkpoint AI-Lab keeps loaded."""
    check_heartmuse(config=heart_config)
    base = heart_config.MODEL_VARIANTS["base"]
    heart_config.MODEL_VARIANTS.clear()
    heart_config.MODEL_VARIANTS["base"] = base
    heart_config.MODEL_VARIANT_LABELS.clear()
    heart_config.MODEL_VARIANT_LABELS["base"] = "HeartMuLa 3B"
    heart_config.DEFAULT_MODEL_VARIANT = "base"
    return heart_config


def launch_heartmuse(backend: HeartMulaBackend, args: argparse.Namespace) -> None:
    """Run HeartMuse's complete UI around AI-Lab's resident pipeline."""
    if not (args.heartmuse_root / "app.py").is_file():
        raise RuntimeError("HeartMuse is not installed")
    os.environ.update({
        "CKPT_DIR": str(args.bundle_root),
        "OUTPUT_DIR": str(args.output_root / "heartmuse"),
        "MODEL_VARIANT": "base", "LAZY_LOAD": "true",
        "HF_HUB_OFFLINE": "1", "SERVER_HOST": PAGE_INTERFACES,
        "SERVER_PORT": str(args.ui_port), "STYLE_TRANSFER": "true",
        "TRANSCRIPTION": "true",
    })
    # HeartMuse's own modules, found through its folder at run time.
    sys.path.insert(0, str(args.heartmuse_root))
    heart_config = _base_only(importlib.import_module("config"))
    # Imported after the variants are narrowed, as before: it may read them.
    generator: Any = importlib.import_module("generator")
    check_heartmuse(generator=generator)
    generator._pipeline = backend.pipeline
    generator._active_lazy_load = True
    generator._active_variant = "base"
    original_generate = generator.generate_music
    def serialized_generate(*values, **options):
        """The page's generate, one at a time with the API's."""
        with backend.lock:
            return original_generate(*values, **options)
    generator.generate_music = serialized_generate
    generator.unload_pipeline = lambda: None
    heartmuse: Any = importlib.import_module("app")
    check_heartmuse(app=heartmuse)
    heartmuse.app.queue(default_concurrency_limit=1).launch(
        server_name=PAGE_INTERFACES, server_port=args.ui_port,
        allowed_paths=[heart_config.OUTPUT_DIR], js=heartmuse.PLAYLIST_JS,
        share=False, inbrowser=False, prevent_thread_lock=True,
        show_error=True)


if __name__ == "__main__":
    main()
