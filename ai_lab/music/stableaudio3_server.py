#!/usr/bin/env python3
"""HTTP host for Stable Audio 3 instrumental music, plus its own web page.

The model is loaded once. The API (for media jobs) and Stability's official
Gradio page (for a person, on port + 10000) share that one loaded model, so
opening the page costs no second copy of the model on the card.
"""
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
    parser.add_argument("--ui-port", type=int, required=True)
    args = parser.parse_args()
    backend = StableAudio3Backend(args.model_path, args.steps, args.cfg_scale)
    launch_native_ui(backend, args.ui_port)
    # Room for the base64 source recordings (two WAVs of up to 25 MiB).
    serve(backend, args.port, max_body_bytes=72 * 1024 * 1024)


def launch_native_ui(backend: StableAudio3Backend, port: int) -> None:
    """Start Stability's own page in the background, on the loaded model."""
    from stable_audio_3.interface.diffusion_cond import create_diffusion_cond_ui

    page = create_diffusion_cond_ui(backend.model, gradio_title="Stable Audio 3")
    page.queue(default_concurrency_limit=1)
    page.launch(server_name="0.0.0.0", server_port=port, share=False,
                inbrowser=False, prevent_thread_lock=True, show_error=True,
                js=getattr(page, "_sao_js", None),
                theme=getattr(page, "_sao_theme", None))


if __name__ == "__main__":
    main()
