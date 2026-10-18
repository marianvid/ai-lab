#!/usr/bin/env python3
"""Isolated HTTP host for YuE2 music generation.

YuE2 runs inside ds-yue-webui ("YuE2 Studio"), which keeps the model loaded
and has its own page. This writes the studio's configuration, starts it, waits
until it answers, and then answers AI-Lab's music contract through it
(`Yue2WebBackend`). Stopping this host stops the studio.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from ai_lab.media.http_host import serve
from ai_lab.music.hosting import (
    PORT,
    UI_PORT,
    Readiness,
    parse,
    stop_on_exit,
    terminate,
    wait_until_ready,
)
from ai_lab.music.yue2_web_backend import Yue2WebBackend
from ai_lab.network import ALL_INTERFACES, LOOPBACK

# How long the studio may take to come up, and to stop.
STUDIO_START_S = 60
STUDIO_STOP_S = 15


def main() -> None:
    """Start YuE2 Studio, then answer music requests through it."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--cot", choices=("full", "melody"), required=True)
    parser.add_argument("--device", choices=("cuda", "mps"), default="cuda")
    args = parse(("--model-path", Path), ("--vae-path", Path), ("--output-root", Path),
                 ("--memory-budget-gib", float), PORT, UI_PORT, ("--webui-root", Path),
                 parser=parser)
    ui_url = f"http://{LOOPBACK}:{args.ui_port}"
    launch_studio(args, ui_url)
    serve(Yue2WebBackend(ui_url, args.model_path.name, args.cot), args.port)


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
         "--host", ALL_INTERFACES, "--port", str(args.ui_port)],
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
        f"host: {ALL_INTERFACES}\n"
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
