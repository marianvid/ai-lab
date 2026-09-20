# Use a model directly

Every configured model has an agent-facing API. The **Models** page also offers
a direct action where AI-Lab has a browser workflow for that task. AI-Lab's
workbench sends requests to the gateway, so a stopped model can be loaded on
the first request. A model in Library still needs a configured entry and a
working engine before it can be used.

| Task | Direct action | Current route |
|---|---|---|
| Text (llama.cpp) | Chat in the engine's own UI when ready; AI-Lab chat while stopped | `/v1/chat/completions` through the gateway |
| Text (vLLM) | AI-Lab chat | `/v1/chat/completions` through the gateway |
| Image generation/editing | Create/Edit with a named ComfyUI profile | AI-Lab image jobs |
| Transcription | Upload audio and read text | `/v1/audio/transcriptions` |
| VAD/diarization | Upload audio and inspect segments/speakers | Task-specific audio routes |
| Transcript alignment | Upload audio and transcript, inspect word timestamps | `/v1/audio/alignments` through the gateway |
| OCR | Upload an image and read recognized text | `/v1/images/ocr` |
| Music (ACE-Step 1.5 XL Turbo) | Music form with player and WAV download | `/v1/audio/music/generations` through the gateway |
| Music (MiniMax Music 3 on Linux) | Music form with instrumental or lyric controls, player and WAV download | `/v1/audio/music/generations` through the gateway |
| Music (YuE2 on Linux) | Music form with required lyrics, editable ABC score, player and WAV/ABC download | `/v1/audio/music/generations` through the gateway |
| Music (HeartMuLa on Linux) | Music form with required lyrics, player and WAV download | `/v1/audio/music/generations` through the gateway |
| Music (Khala on macOS) | Music form with length bucket, player and WAV download | `/v1/audio/music/generations` through the gateway |
| Speech (Qwen3-TTS VoiceDesign/CustomVoice, Kokoro, VoxCPM, Higgs on Linux) | Speak form with engine-specific controls, player and WAV download | `/v1/audio/speech/generations` through the gateway |
| Other installed music, speech, alignment and video models | No direct action until an engine and job contract are configured | Library only |

The new chat workbench is deliberately small: it keeps the current
conversation in the page and sends one request at a time. llama.cpp's own UI
remains available for its richer chat features. [Open WebUI](https://docs.openwebui.com/getting-started/quick-start/connect-a-provider/starting-with-vllm/)
can connect to vLLM or AI-Lab's OpenAI-compatible gateway if a richer shared
chat workspace is wanted. vLLM itself documents an
[OpenAI-compatible server](https://docs.vllm.ai/en/latest/serving/online_serving/openai_compatible_server/)
and separate UI examples, rather than a built-in chat page like
[llama.cpp](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md).

ComfyUI has its own workflow interface, but AI-Lab currently runs named
workflows through an adapter. Opening ComfyUI separately would need a
supervised, access-controlled route that respects AI-Lab's model scheduling;
the existing adapter intentionally does not expose that native UI. The
[ACE-Step project](https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/UI_SUPPORT.md)
offers a full Gradio interface for music experiments and a separate REST API.
[Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) provides the `qwen-tts-demo`
web interface and documents an inference path through vLLM-Omni. These are
viable direct experimentation surfaces for the corresponding models, but a
standalone server would compete with AI-Lab for accelerator memory. AI-Lab
provides controlled ACE-Step, HeartMuLa, YuE2, MiniMax Music 3, Khala, Qwen3-TTS, Kokoro, VoxCPM and Higgs adapters for configured checkpoints
on Linux and macOS. Browser and agent requests use the same gateway lease,
held until the WAV is returned. Add or replace a checkpoint in the private
configuration; the adapter reads its mode from that configuration, not its
version or instance name. Other library models still require their own
lifecycle adapter, result contract and resource checks before direct use.
VoxCPM supports voice descriptions in the Speak form. Reference-audio cloning
still needs an upload field and request contract; it is not offered by this form.
Higgs supports inline style controls in the speech text; its installed
SGLang-Omni worker is loaded and stopped with the AI-Lab instance.
MiniMax Music 3 runs a configured ComfyUI audio workflow inside a supervised
AI-Lab worker. Its checkpoint components and workflow are selected through
private configuration. [ComfyUI documentation](https://docs.comfy.org/tutorials/audio/minimax/minimax-music-3)
shows the native workflow controls.
YuE2 returns an ABC composition alongside its audio. The Music form lets you edit
that score and regenerate; it reports when the model marks a song as truncated.
YuE2 determines song length from the composition, so its form has no duration
control. [YuE2 generation documentation](https://github.com/multimodal-art-projection/YuE/blob/main/docs/generation.md)
describes the supported score-editing workflow.
HeartMuLa requires lyrics; the Music form marks that field as required when this
engine is selected. Its installed checkpoint and supporting codec/tokenizer are
selected through private configuration.
Khala accepts `length_bucket` instead of seconds. Its output duration varies;
the browser displays that control using the runtime's actual unit.

The actual Linux and Mac entries, paths and workflow profiles belong to the
private `opts` repository. This public page describes behavior without
publishing installation secrets.
