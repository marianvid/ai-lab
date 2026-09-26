#!/usr/bin/env python3
"""Isolated HTTP host for configurable speech backends."""
from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from ai_lab.speech.contract import MAX_REQUEST_BYTES, PATH, validate_payload
from ai_lab.speech.higgs_local_backend import HiggsLocalBackend
from ai_lab.speech.kokoro_backend import KokoroBackend
from ai_lab.speech.qwen import MODES, QwenTtsBackend
from ai_lab.speech.voxcpm_backend import VoxCpmBackend


class Handler(BaseHTTPRequestHandler):
    backend = None

    def do_GET(self):
        if self.path == "/health":
            self._json(200, {"status": "ok"})
        else:
            self._json(404, {"error": {"message": "not found"}})

    def do_POST(self):
        if self.path != PATH:
            self._json(404, {"error": {"message": "not found"}})
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= MAX_REQUEST_BYTES:
                raise ValueError("speech request is empty or too large")
            body = json.loads(self.rfile.read(size))
            if not isinstance(body, dict):
                raise ValueError("speech request must be an object")
            self._json(200, self.backend.generate(body))
        except (ValueError, TypeError) as error:
            self._json(400, {"error": {"message": str(error)}})
        except Exception as error:
            self._json(500, {"error": {"message": str(error)}})

    def _json(self, status: int, body: dict):
        payload = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("qwen", "kokoro", "voxcpm", "higgs"), default="qwen")
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
    if args.backend == "qwen":
        if not args.mode:
            parser.error("Qwen speech requires --mode")
        Handler.backend = QwenTtsBackend(args.model_path, args.mode)
        if args.ui_port is None:
            parser.error("Qwen speech requires --ui-port")
        launch_native_qwen_ui(Handler.backend, args.model_path, args.ui_port)
    elif args.backend == "kokoro":
        if not all((args.language_code, args.default_voice, args.repo_id)):
            parser.error("Kokoro requires language, voice and repo settings")
        Handler.backend = KokoroBackend(
            args.model_path, args.language_code, args.default_voice,
            args.repo_id)
        if args.ui_port is None:
            parser.error("Kokoro speech requires --ui-port")
        launch_native_kokoro_ui(Handler.backend, args.model_path.parent,
                                args.ui_port)
    elif args.backend == "higgs":
        Handler.backend = HiggsLocalBackend(
            args.model_path, args.max_batch, args.batch_window_ms)
        if args.ui_port is None:
            parser.error("Higgs speech requires --ui-port")
        # The same upstream playground page Linux shows, answered by the
        # model this process already holds (see higgs_playground.py).
        from ai_lab.speech import higgs_playground

        higgs_playground.serve(Handler.backend, args.ui_port)
    else:
        Handler.backend = VoxCpmBackend(
            args.model_path, args.cfg_value, args.inference_timesteps)
        if args.ui_port is None:
            parser.error("VoxCPM speech requires --ui-port")
        launch_native_voxcpm_ui(Handler.backend, args.model_path, args.ui_port)
    ThreadingHTTPServer(("0.0.0.0", args.port), Handler).serve_forever()


def launch_native_qwen_ui(backend: QwenTtsBackend, model_path: Path,
                          port: int) -> None:
    """Use Qwen's full Gradio editor without loading the checkpoint twice."""
    from qwen_tts.cli.demo import build_demo

    demo = build_demo(backend.model, str(model_path), {})
    demo.queue(default_concurrency_limit=1)
    demo.launch(server_name="0.0.0.0", server_port=port, share=False,
                inbrowser=False, prevent_thread_lock=True, show_error=True)


def launch_native_voxcpm_ui(backend: VoxCpmBackend, model_path: Path,
                            port: int) -> None:
    """Use the full upstream VoxCPM editor and its already-loaded model."""
    from ai_lab.native_ui.voxcpm_upstream import app as upstream

    demo = upstream.VoxCPMDemo(model_id=str(model_path))
    demo.voxcpm_model = backend.model
    interface = upstream.create_demo_interface(demo)
    interface.queue(max_size=10, default_concurrency_limit=1).launch(
        server_name="0.0.0.0", server_port=port, show_error=True,
        i18n=upstream.I18N, theme=upstream._APP_THEME,
        css=upstream._CUSTOM_CSS, share=False, inbrowser=False,
        prevent_thread_lock=True)


def launch_native_kokoro_ui(backend: KokoroBackend, model_path: Path,
                            port: int) -> None:
    """Use Kokoro's full upstream editor with the resident AI-Lab model."""
    from ai_lab.native_ui.kokoro_upstream.app import create_demo

    interface = create_demo(backend, model_path)
    interface.queue(max_size=10, default_concurrency_limit=1).launch(
        server_name="0.0.0.0", server_port=port, show_error=True,
        share=False, inbrowser=False, prevent_thread_lock=True)


if __name__ == "__main__":
    main()
