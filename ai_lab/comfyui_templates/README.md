These native editors are adapted from the [official ComfyUI workflow
templates](https://github.com/Comfy-Org/workflow_templates) (MIT). Each uses
AI-Lab's installed checkpoint, text encoder and VAE; the automatic benchmark
API workflows remain separate and unchanged.

| AI-Lab editor | Upstream template |
|---|---|
| `audio_minimax_music_3.json` | `audio_minimax_music_3.json` |
| `image_qwen_2512_nvfp4.json` | `image_qwen_Image_2512.json` |
| `image_qwen_edit_2511_fp8mixed.json` | `image_qwen_image_edit_2511.json` |
| `image_flux2_dev_q8_{linux,macos}.json` | `image_flux2_text_to_image.json` |
| `image_flux2_klein_4b_distilled.json` | `image_flux2_klein_text_to_image.json` (distilled branch only) |
| `video_minimax_h3_fp8.json` | `video_minimax_h3_i2v.json` |
| `video_ltx2_5_nvfp4.json` | `video_ltx2_5_i2v.json` |

All upstream files are in the repository's `templates/` directory. The
model-specific editor is loaded only for that model's AI-Lab instance. A
template bypasses optional LoRA/enhancer weights that are not installed.
Keep the upstream fast-step/LoRA switches off: without the corresponding
LoRA, those short schedules do not represent the advertised fast variant.
A manual image/video input must be supplied before running a
workflow that starts with `LoadImage`. The templates do not download extra
checkpoints merely to reproduce optional features shown in upstream examples.
