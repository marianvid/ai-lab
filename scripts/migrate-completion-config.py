#!/usr/bin/env python3
"""Idempotently add the completion-phase Linux configuration.

Existing instances and operator settings win. This only adds missing roots,
repositories, engine paths and image-job defaults, and writes atomically.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path


def merge(config: dict) -> dict:
    roots = config.setdefault("model_roots", [])
    if not any(item.get("id") == "core" for item in roots):
        roots.insert(0, {"id": "core", "name": "Core",
                         "path": config.get("models_root", "/models"),
                         "enabled": True, "writable": True})
    if not any(item.get("id") == "benchmark" for item in roots):
        roots.append({"id": "benchmark", "name": "Benchmark",
                      "path": "/test_models", "enabled": True,
                      "writable": True})
    config.setdefault("download_root", "benchmark")

    repositories = config.setdefault("repositories", [])
    additions = (
        {"id": "images-paddleocr", "name": "PaddleOCR", "format": "paddleocr",
         "task": "ocr", "subpath": "images/ocr"},
        {"id": "images-comfyui-generation", "name": "ComfyUI generation models",
         "format": "comfyui", "task": "image-generation",
         "subpath": "images/generation"},
        {"id": "images-comfyui-edit", "name": "ComfyUI editing models",
         "format": "comfyui", "task": "image-edit", "subpath": "images/edit"},
    )
    known = {item.get("id") for item in repositories}
    repositories.extend(item for item in additions if item["id"] not in known)

    engines = config.setdefault("engines", {})
    engines.setdefault("paddleocr", {
        "binary": "/opt/ai/paddleocr/current/bin/python",
        "server": "/opt/ai-lab/ai_lab/images/server.py",
        "source": {"package": "paddleocr", "modules": ["paddleocr", "paddle"],
                   "requirements": [
                       "paddlepaddle-gpu @ https://paddle-whl.cdn.bcebos.com/stable/cu130/paddlepaddle-gpu/paddlepaddle_gpu-3.3.0-cp312-cp312-linux_x86_64.whl"],
                   "minimum_versions": {"paddlepaddle-gpu": "3.3.0"},
                   "python": "/opt/ai/python/cpython-3.12.14-linux-x86_64-gnu/bin/python3.12",
                   "root": "/opt/ai/paddleocr", "uv": "/usr/local/bin/uv"},
    })
    paddle_source = engines["paddleocr"].setdefault("source", {})
    paddle_source.setdefault("package", "paddleocr")
    paddle_source.setdefault("modules", ["paddleocr", "paddle"])
    # These values deliberately replace the first completion migration's
    # generic PyPI source, which can resolve the obsolete GPU 2.6 wheel.
    paddle_source["requirements"] = [
        "paddlepaddle-gpu @ https://paddle-whl.cdn.bcebos.com/stable/cu130/"
        "paddlepaddle-gpu/paddlepaddle_gpu-3.3.0-cp312-cp312-linux_x86_64.whl"]
    paddle_source["minimum_versions"] = {"paddlepaddle-gpu": "3.3.0"}
    paddle_source.pop("pip_args", None)
    paddle_source.setdefault(
        "python", "/opt/ai/python/cpython-3.12.14-linux-x86_64-gnu/bin/python3.12")
    paddle_source.setdefault("root", "/opt/ai/paddleocr")
    paddle_source.setdefault("uv", "/usr/local/bin/uv")
    engines.setdefault("comfyui", {
        "binary": "/opt/ai/comfyui/current/.venv/bin/python",
        "server": "/opt/ai-lab/ai_lab/images/comfyui_server.py",
        "comfyui": "/opt/ai/comfyui/current/ComfyUI/main.py",
        "model_paths": ["/models", "/test_models"],
    })
    comfy = engines["comfyui"]
    comfy["binary"] = "/opt/ai/comfyui/current/.venv/bin/python"
    comfy["comfyui"] = "/opt/ai/comfyui/current/ComfyUI/main.py"
    comfy.setdefault("server", "/opt/ai-lab/ai_lab/images/comfyui_server.py")
    comfy.setdefault("model_paths", ["/models", "/test_models"])
    comfy.setdefault("source", {}).update({"kind": "git-app", "root": "/opt/ai/comfyui",
        "repository": "https://github.com/Comfy-Org/ComfyUI.git", "application": "ComfyUI",
        "python": "/opt/ai/python/cpython-3.12.14-linux-x86_64-gnu/bin/python3.12",
        "uv": "/usr/local/bin/uv", "requires_cuda": True, "verify_module": "server"})
    images = config.setdefault("images", {
        "workflow_root": "/etc/ai-lab/image-workflows",
        "state_dir": "/var/lib/ai-lab/image-jobs",
        "client_wait_s": 1800,
        "result_ttl_s": 86400,
        "profiles": {},
    })
    images.setdefault("profiles", {}).setdefault("sd15-smoke", {
        "description": "Controlled SD 1.5 generation smoke profile",
        "version": "1",
        "task": "generation",
        "model": "image-smoke",
        "workflow": "sd15-generation-v1.json",
        "inputs": {
            "prompt": ["2", "text"],
            "negative_prompt": ["3", "text"],
            "seed": ["5", "seed"],
            "steps": ["5", "steps"],
            "cfg_scale": ["5", "cfg"],
            "width": ["4", "width"],
            "height": ["4", "height"],
        },
    })
    images["profiles"].setdefault("sd15-edit-smoke", {
        "description": "Controlled SD 1.5 image-edit smoke profile",
        "version": "1",
        "task": "edit",
        "model": "image-smoke",
        "workflow": "sd15-edit-v1.json",
        "inputs": {
            "prompt": ["2", "text"],
            "negative_prompt": ["3", "text"],
            "image": ["4", "image"],
            "seed": ["6", "seed"],
            "steps": ["6", "steps"],
            "cfg_scale": ["6", "cfg"],
        },
    })
    images["profiles"].setdefault("qwen-image-benchmark", {
        "description": "Qwen Image NVFP4 deterministic benchmark profile",
        "version": "1",
        "task": "generation",
        "model": "image-qwen",
        "workflow": "qwen-image-generation-v1.json",
        "inputs": {
            "prompt": ["5", "text"],
            "negative_prompt": ["6", "text"],
            "seed": ["8", "seed"],
            "steps": ["8", "steps"],
            "cfg_scale": ["8", "cfg"],
            "width": ["7", "width"],
            "height": ["7", "height"],
        },
    })
    images["profiles"].setdefault("flux2-benchmark", {
        "description": "FLUX.2 Dev FP8 deterministic benchmark profile",
        "version": "1",
        "task": "generation",
        "model": "image-flux2",
        "workflow": "flux2-generation-v1.json",
        "inputs": {
            "prompt": ["4", "text"],
            "seed": ["10", "noise_seed"],
            "steps": ["8", "steps"],
            "width": ["8", "width"],
            "height": ["8", "height"],
        },
    })
    images["profiles"].setdefault("qwen-edit-benchmark", {
        "description": "Qwen Image Edit 2511 FP8 deterministic benchmark profile",
        "version": "1",
        "task": "edit",
        "model": "image-qwen-edit",
        "workflow": "qwen-image-edit-v1.json",
        "inputs": {
            "image": ["1", "image"],
            "prompt": ["7", "prompt"],
            "seed": ["10", "seed"],
            "steps": ["10", "steps"],
            "cfg_scale": ["10", "cfg"],
        },
    })
    downloads = config.setdefault("downloads", {})
    bundles = downloads.setdefault("bundles", [])
    bundle_additions = (
        {
            "name": "qwen-image-2512-nvfp4",
            "repo": "Comfy-Org/Qwen-Image_ComfyUI",
            "format": "comfyui",
            "task": "image-generation",
            "components": [
                {"role": "diffusion_model", "path":
                 "split_files/diffusion_models/qwen_image_nvfp4.safetensors"},
                {"role": "text_encoder", "path":
                 "split_files/text_encoders/qwen_2.5_vl_7b_nvfp4.safetensors"},
                {"role": "vae", "path":
                 "split_files/vae/qwen_image_vae.safetensors"},
            ],
        },
        {
            "name": "qwen-image-edit-2511-fp8mixed",
            "repo": "Comfy-Org/Qwen-Image-Edit_ComfyUI",
            "format": "comfyui",
            "task": "image-edit",
            "components": [
                {"role": "diffusion_model", "path":
                 "split_files/diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors"},
                {"role": "text_encoder", "repo": "Comfy-Org/Qwen-Image_ComfyUI",
                 "path": "split_files/text_encoders/qwen_2.5_vl_7b_nvfp4.safetensors"},
                {"role": "vae", "repo": "Comfy-Org/Qwen-Image_ComfyUI",
                 "path": "split_files/vae/qwen_image_vae.safetensors"},
            ],
        },
        {
            "name": "flux2-dev-fp8mixed",
            "repo": "Comfy-Org/flux2-dev",
            "format": "comfyui",
            "task": "image-generation",
            "components": [
                {"role": "diffusion_model", "path":
                 "split_files/diffusion_models/flux2_dev_fp8mixed.safetensors"},
                {"role": "text_encoder", "path":
                 "split_files/text_encoders/mistral_3_small_flux2_fp4_mixed.safetensors"},
                {"role": "vae", "path": "split_files/vae/flux2-vae.safetensors"},
            ],
        },
    )
    known_bundles = {item.get("name") for item in bundles}
    bundles.extend(item for item in bundle_additions
                   if item["name"] not in known_bundles)
    gateway = config.setdefault("gateway", {})
    gateway.setdefault("max_upload_bytes", 26214400)
    gateway.setdefault("max_upload_pixels", 50000000)
    gateway.setdefault("max_upload_dimension", 8000)
    gateway.setdefault("task_timeouts", {}).setdefault(
        "ocr", {"first_byte_s": 60, "between_bytes_s": 30})
    gateway["task_timeouts"].setdefault(
        "image-generation", {"first_byte_s": 1800, "between_bytes_s": 60})
    gateway["task_timeouts"].setdefault(
        "image-edit", {"first_byte_s": 1800, "between_bytes_s": 60})
    return config


def main() -> None:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "/etc/ai-lab/config.json")
    payload = merge(json.loads(path.read_text()))
    backup = path.with_suffix(path.suffix + ".pre-completion")
    if not backup.exists():
        shutil.copy2(path, backup)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.chown(temporary, path.stat().st_uid, path.stat().st_gid)
    os.chmod(temporary, path.stat().st_mode & 0o777)
    os.replace(temporary, path)


if __name__ == "__main__":
    main()
