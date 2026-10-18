#!/usr/bin/env python3
"""HTTP host for Stable Audio 3 instrumental music, plus its own web page.

The model is loaded once. The API (for media jobs) and Stability's official
Gradio page (for a person, on port + 10000) share that one loaded model, so
opening the page costs no second copy of the model on the card.
"""
from __future__ import annotations

from pathlib import Path

from ai_lab.engine_host.http_host import serve
from ai_lab.music.hosting import PORT, UI_PORT, parse
from ai_lab.music.backends.stableaudio3_backend import StableAudio3Backend
from ai_lab.network import ALL_INTERFACES


# Room for the base64 source recordings (two WAVs of up to 25 MiB).
MAX_BODY_BYTES = 72 * 1024 * 1024


def main() -> None:
    """Load Stable Audio 3 and its own page, then answer music requests."""
    args = parse(("--model-path", Path), ("--steps", int), ("--cfg-scale", float),
                 PORT, UI_PORT)
    backend = StableAudio3Backend(args.model_path, args.steps, args.cfg_scale)
    launch_native_ui(backend, args.ui_port)
    serve(backend, args.port, max_body_bytes=MAX_BODY_BYTES)


def launch_native_ui(backend: StableAudio3Backend, port: int) -> None:
    """Start Stability's own page in the background, on the loaded model."""
    from stable_audio_3.interface.diffusion_cond import create_diffusion_cond_ui

    page = create_diffusion_cond_ui(backend.model, gradio_title="Stable Audio 3")
    page.queue(default_concurrency_limit=1)
    page.launch(server_name=ALL_INTERFACES, server_port=port, share=False,
                inbrowser=False, prevent_thread_lock=True, show_error=True,
                js=getattr(page, "_sao_js", None),
                theme=getattr(page, "_sao_theme", None))


if __name__ == "__main__":
    main()
