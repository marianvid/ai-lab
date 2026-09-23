#!/usr/bin/env python3
"""Isolated HTTP host for YuE2 music generation."""
from __future__ import annotations

import argparse
import atexit
import json
import os
import signal
import subprocess
import time
import urllib.request
from pathlib import Path

from ai_lab.media.http_host import serve
from ai_lab.music.yue2_web_backend import Yue2WebBackend


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--vae-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cot", choices=("full", "melody"), required=True)
    parser.add_argument("--memory-budget-gib", type=float, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--ui-port", type=int, required=True)
    parser.add_argument("--webui-root", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "mps"), default="cuda")
    args = parser.parse_args()
    ui_url = f"http://127.0.0.1:{args.ui_port}"
    launch_studio(args, ui_url)
    backend = Yue2WebBackend(ui_url, args.model_path.name, args.cot)
    serve(backend, args.port)


def studio_env(device: str) -> dict[str, str]:
    """Offline always; on Metal, let an operator PyTorch lacks run on the CPU
    instead of stopping the song, as the measured Mac benchmark did."""
    env = {**os.environ, "HF_HUB_OFFLINE": "1"}
    if device == "mps":
        env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    return env


def launch_studio(args, ui_url: str) -> None:
    if not (args.webui_root / "server" / "__main__.py").is_file():
        raise RuntimeError("ds-yue-webui is not installed")
    data_dir = args.output_root / "yue2-studio"
    data_dir.mkdir(parents=True, exist_ok=True)
    config = data_dir / "config.yaml"
    config.write_text(
        "host: 0.0.0.0\n"
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
        f"  python: {os.sys.executable}\n  python_yue2: {os.sys.executable}\n"
        f"  python_sheetsage2: {os.sys.executable}\n")
    child = subprocess.Popen(
        [os.sys.executable, "-m", "server", "--config", str(config),
         "--host", "0.0.0.0", "--port", str(args.ui_port)],
        cwd=args.webui_root, env=studio_env(args.device))

    def stop() -> None:
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                child.kill()
    atexit.register(stop)
    def terminate(_signum, _frame):
        stop()
        raise SystemExit(0)
    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        if child.poll() is not None:
            raise RuntimeError("YuE2 Studio exited during startup")
        try:
            with urllib.request.urlopen(ui_url + "/api/config", timeout=2) as response:
                json.load(response)
                return
        except Exception:
            time.sleep(1)
    raise RuntimeError("YuE2 Studio did not become ready")


if __name__ == "__main__":
    main()
