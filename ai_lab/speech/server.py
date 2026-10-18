#!/usr/bin/env python3
"""Isolated HTTP host for configurable speech backends.

One backend per process (Qwen TTS, Kokoro, VoxCPM or Higgs on a Mac), with
the backend's own editor on `--ui-port` and AI-Lab's speech JSON contract
(`speech.contract`) on `--port`, answered by `media.http_host`.
"""
from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

from ai_lab.media.http_host import Generator, serve
from ai_lab.speech.contract import MAX_REQUEST_BYTES, PATH
from ai_lab.speech.higgs_local_backend import HiggsLocalBackend
from ai_lab.speech.kokoro_backend import KokoroBackend
from ai_lab.speech.qwen import MODES, QwenTtsBackend
from ai_lab.speech.voxcpm_backend import VoxCpmBackend
from ai_lab.network import ALL_INTERFACES


def main() -> None:
    """Load the speech backend named on the command line, its own editor, and answer."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=tuple(_BUILDERS), default="qwen")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--mode", choices=sorted(MODES))
    parser.add_argument("--language-code")
    parser.add_argument("--default-voice")
    parser.add_argument("--repo-id")
    parser.add_argument("--cfg-value", type=float, default=2.0)
    parser.add_argument("--inference-timesteps", type=int, default=10)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--ui-port", type=int)
    parser.add_argument("--max-batch", type=int, default=1)
    parser.add_argument("--batch-window-ms", type=int, default=100)
    args = parser.parse_args()
    if args.ui_port is None:
        parser.error(f"{args.backend} speech requires --ui-port")
    backend = _BUILDERS[args.backend](args, parser)
    serve(backend, args.port, max_body_bytes=MAX_REQUEST_BYTES, request_path=PATH,
          noun="speech")


def _qwen(args, parser: argparse.ArgumentParser) -> QwenTtsBackend:
    if not args.mode:
        parser.error("Qwen speech requires --mode")
    backend = QwenTtsBackend(args.model_path, args.mode)
    launch_native_qwen_ui(backend, args.model_path, args.ui_port)
    return backend


def _kokoro(args, parser: argparse.ArgumentParser) -> KokoroBackend:
    if not all((args.language_code, args.default_voice, args.repo_id)):
        parser.error("Kokoro requires language, voice and repo settings")
    backend = KokoroBackend(args.model_path, args.language_code, args.default_voice,
                            args.repo_id)
    launch_native_kokoro_ui(backend, args.model_path.parent, args.ui_port)
    return backend


def _higgs(args, _parser: argparse.ArgumentParser) -> HiggsLocalBackend:
    backend = HiggsLocalBackend(args.model_path, args.max_batch, args.batch_window_ms)
    # The same upstream playground page Linux shows, answered by the
    # model this process already holds (see higgs_playground.py).
    from ai_lab.speech import higgs_playground

    higgs_playground.serve(backend, args.ui_port)
    return backend


def _voxcpm(args, _parser: argparse.ArgumentParser) -> VoxCpmBackend:
    backend = VoxCpmBackend(args.model_path, args.cfg_value, args.inference_timesteps)
    launch_native_voxcpm_ui(backend, args.model_path, args.ui_port)
    return backend


# How each backend is built and given its own editor, by its `--backend` name.
_BUILDERS: dict[str, Callable[[argparse.Namespace, argparse.ArgumentParser], Generator]] = {
    "qwen": _qwen, "kokoro": _kokoro, "voxcpm": _voxcpm, "higgs": _higgs}


def launch_native_qwen_ui(backend: QwenTtsBackend, model_path: Path,
                          port: int) -> None:
    """Use Qwen's full Gradio editor without loading the checkpoint twice."""
    from qwen_tts.cli.demo import build_demo

    demo = build_demo(backend.model, str(model_path), {})
    demo.queue(default_concurrency_limit=1)
    demo.launch(server_name=ALL_INTERFACES, server_port=port, share=False,
                inbrowser=False, prevent_thread_lock=True, show_error=True)


def launch_native_voxcpm_ui(backend: VoxCpmBackend, model_path: Path,
                            port: int) -> None:
    """Use the full upstream VoxCPM editor and its already-loaded model."""
    from ai_lab.vendor.voxcpm_upstream import app as upstream

    demo = upstream.VoxCPMDemo(model_id=str(model_path))
    demo.voxcpm_model = backend.model
    interface = upstream.create_demo_interface(demo)
    interface.queue(max_size=10, default_concurrency_limit=1).launch(
        server_name=ALL_INTERFACES, server_port=port, show_error=True,
        i18n=upstream.I18N, theme=upstream._APP_THEME,
        css=upstream._CUSTOM_CSS, share=False, inbrowser=False,
        prevent_thread_lock=True)


def launch_native_kokoro_ui(backend: KokoroBackend, model_path: Path,
                            port: int) -> None:
    """Use Kokoro's full upstream editor with the resident AI-Lab model."""
    from ai_lab.vendor.kokoro_upstream.app import create_demo

    interface = create_demo(backend, model_path)
    interface.queue(max_size=10, default_concurrency_limit=1).launch(
        server_name=ALL_INTERFACES, server_port=port, show_error=True,
        share=False, inbrowser=False, prevent_thread_lock=True)


if __name__ == "__main__":
    main()
