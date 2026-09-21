"""Model-specific native ComfyUI editors, independent of benchmark graphs."""

import sys
from pathlib import Path


_TEMPLATES = {
    "qwen-image-2512-nvfp4": "image_qwen_2512_nvfp4.json",
    "qwen-image-edit-2511-fp8mixed": "image_qwen_edit_2511_fp8mixed.json",
    "flux2-klein-4b-bf16": "image_flux2_klein_4b_distilled.json",
    "minimax-h3": "video_minimax_h3_fp8.json",
    "ltx-2.5-nvfp4": "video_ltx2_5_nvfp4.json",
    "minimax-music3": "audio_minimax_music_3.json",
}


def for_model(name: str, platform: str | None = None) -> Path | None:
    if name == "flux2-dev-q8-0":
        filename = ("image_flux2_dev_q8_macos.json" if
                    (platform or sys.platform) == "darwin" else
                    "image_flux2_dev_q8_linux.json")
    else:
        filename = _TEMPLATES.get(name)
    return Path(__file__).resolve().parent / filename if filename else None
