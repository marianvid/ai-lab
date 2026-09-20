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
| OCR | Upload an image and read recognized text | `/v1/images/ocr` |
| Music (ACE-Step 1.5 XL Turbo) | Music form with player and WAV download | `/v1/audio/music/generations` through the gateway |
| Other music and TTS | No AI-Lab engine and job contract yet | Future work |

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
standalone server would compete with AI-Lab for accelerator memory. AI-Lab now provides one controlled ACE-Step adapter for the selected XL Turbo
model on Linux and macOS. It serves both the browser music form and an agent
request, with AI-Lab holding a model lease until the WAV is returned. The
remaining models still require their own lifecycle adapter, result contract
and resource checks. A library label alone is insufficient. The direct UI and those
contracts should be designed before the large gateway and operations classes
are split.

The actual Linux and Mac entries, paths and workflow profiles belong to the
private `opts` repository. This public page describes behavior without
publishing installation secrets.
