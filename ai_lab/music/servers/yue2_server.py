#!/usr/bin/env python3
"""Isolated HTTP host for YuE2 music generation.

YuE2 runs inside ds-yue-webui ("YuE2 Studio"), which keeps the model loaded
and has its own page. This writes the studio's configuration, starts it, waits
until it answers, and then answers AI-Lab's music contract through it
(`Yue2WebBackend`). Stopping this host stops the studio.

Besides the song call it answers `POST /v1/audio/music/window-generations`
(one window of an earlier song regenerated, when every candidate is done) and
`DELETE /v1/audio/music/window-generations/{request_id}` (stop one). A refused
window request is a 422, a `request_id` already running a 409, an unknown one
a 404 (`docs/music.md`).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Callable
from http import HTTPStatus
from http.server import ThreadingHTTPServer
from pathlib import Path

from ai_lab.engine_host.http_host import GenerationHost
from ai_lab.music.hosting import (
    PORT,
    UI_PORT,
    Readiness,
    parse,
    stop_on_exit,
    terminate,
    wait_until_ready,
)
from ai_lab.music.backends.yue2_web_backend import (
    WindowActiveError,
    WindowRefusedError,
    WindowSettings,
    Yue2WebBackend,
)
from ai_lab.network import LOOPBACK, PAGE_INTERFACES

# How long the studio may take to come up, and to stop.
STUDIO_START_S = 60
STUDIO_STOP_S = 15
WINDOW_PATH = "/v1/audio/music/window-generations"
# Each refusal of a window call and the status it is answered with; anything else is a 500.
WINDOW_STATUS: tuple[tuple[type[Exception], HTTPStatus], ...] = (
    (WindowRefusedError, HTTPStatus.UNPROCESSABLE_ENTITY),
    (WindowActiveError, HTTPStatus.CONFLICT),
    (KeyError, HTTPStatus.NOT_FOUND),
    (ValueError, HTTPStatus.BAD_REQUEST),
    (TypeError, HTTPStatus.BAD_REQUEST),
)


class Yue2Host(GenerationHost):
    """`GenerationHost` with the window call and its cancel (`Yue2WebBackend`)."""

    backend: Yue2WebBackend

    def do_POST(self) -> None:  # noqa: N802 - the name http.server calls
        """The window call, or the ordinary music call."""
        if self.path != WINDOW_PATH:
            super().do_POST()
            return
        backend = self.backend
        self._answer(lambda: backend.generate_window(self._body()))

    def do_DELETE(self) -> None:  # noqa: N802 - the name http.server calls
        """Cancel the window call named by the last part of the path."""
        prefix = WINDOW_PATH + "/"
        request_id = self.path[len(prefix):] if self.path.startswith(prefix) else ""
        if not request_id or "/" in request_id:
            self._not_found()
            return
        backend = self.backend
        self._answer(lambda: backend.cancel_window(request_id))

    def _answer(self, work: Callable[[], dict]) -> None:
        try:
            self._json(HTTPStatus.OK, work())
        except Exception as error:
            status = next((status for kind, status in WINDOW_STATUS if isinstance(error, kind)),
                          HTTPStatus.INTERNAL_SERVER_ERROR)
            message = error.args[0] if isinstance(error, KeyError) and error.args else str(error)
            self._json(status, {"error": {"message": message}})


def serve(backend: Yue2WebBackend, port: int) -> None:
    """Answer the song and window calls on this machine only (loopback) until stopped."""
    handler = type("BoundYue2Host", (Yue2Host,), {"backend": backend})
    ThreadingHTTPServer((LOOPBACK, port), handler).serve_forever()


def main() -> None:
    """Start YuE2 Studio, then answer music requests through it."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--cot", choices=("full", "melody"), required=True)
    parser.add_argument("--device", choices=("cuda", "mps"), default="cuda")
    # The instance's `window_*` settings as JSON (`Yue2Engine.plan`); without them
    # window calls are refused rather than run on numbers nobody chose.
    parser.add_argument("--window-settings", type=json.loads, default=None)
    args = parse(("--model-path", Path), ("--vae-path", Path), ("--output-root", Path),
                 ("--memory-budget-gib", float), PORT, UI_PORT, ("--webui-root", Path),
                 parser=parser)
    ui_url = f"http://{LOOPBACK}:{args.ui_port}"
    launch_studio(args, ui_url)
    window = (WindowSettings.from_params(args.window_settings)
              if args.window_settings is not None else None)
    serve(Yue2WebBackend(ui_url, args.model_path.name, args.cot, window=window), args.port)


def studio_env(device: str) -> dict[str, str]:
    """Offline always; on Metal, let an operator PyTorch lacks run on the CPU
    instead of stopping the song, as the measured Mac benchmark did."""
    env = {**os.environ, "HF_HUB_OFFLINE": "1"}
    if device == "mps":
        env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    return env


def launch_studio(args: argparse.Namespace, ui_url: str) -> None:
    """Start YuE2 Studio with this instance's settings and wait until it answers."""
    if not (args.webui_root / "server" / "__main__.py").is_file():
        raise RuntimeError("ds-yue-webui is not installed")
    config = studio_config(args)
    child = subprocess.Popen(
        [sys.executable, "-m", "server", "--config", str(config),
         "--host", PAGE_INTERFACES, "--port", str(args.ui_port)],
        cwd=args.webui_root, env=studio_env(args.device))
    stop_on_exit(lambda: terminate(child, STUDIO_STOP_S))
    wait_until_ready(Readiness(ui_url + "/api/config", STUDIO_START_S, _answers_json),
                     "YuE2 Studio", child)


def _answers_json(response) -> bool:
    """Ready once the studio's configuration reads as JSON."""
    json.load(response)
    return True


def studio_config(args: argparse.Namespace) -> Path:
    """The studio's `config.yaml`, written under the output folder; its path."""
    data_dir = args.output_root / "yue2-studio"
    data_dir.mkdir(parents=True, exist_ok=True)
    config = data_dir / "config.yaml"
    python = sys.executable
    config.write_text(
        f"host: {PAGE_INTERFACES}\n"
        f"port: {args.ui_port}\n"
        f"data_dir: {data_dir}\n"
        "residency: always\nrelease_idle_minutes: 0\noffline: true\n"
        f"memory_budget_gib: {args.memory_budget_gib}\n"
        "yue2:\n"
        f"  model: {args.model_path}\n  vae: {args.vae_path}\n"
        f"  vae_legacy: {args.vae_path}\n  device: {args.device}\n"
        "  backend: torch\n  quantization: none\n  offload_ar: true\n"
        f"sheetsage2:\n  model: auto\n  device: {args.device}\n  dtype: bf16\n"
        "worker:\n"
        f"  python: {python}\n  python_yue2: {python}\n"
        f"  python_sheetsage2: {python}\n")
    return config


if __name__ == "__main__":
    main()
