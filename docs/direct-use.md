# Use a loaded model directly

The Models page links to a native interface only when the configured instance
is ready. A stopped instance shows a disabled action; it starts no UI process.

| Engine | Action | Interface |
|---|---|---|
| llama.cpp | Chat | The engine's own chat page |
| ComfyUI image/edit | Create / Edit | Native ComfyUI with a model-specific template adapted to installed weights |
| ComfyUI video | Video | Native ComfyUI with a model-specific image-to-video template adapted to installed weights |
| ComfyUI music | Music | Native ComfyUI with the MiniMax Music 3 template adapted to INT8 |
| ACE-Step 1.5 | Music | Upstream Gradio playground using the already-loaded AI-Lab model |
| Qwen3-TTS VoiceDesign / CustomVoice | Speak | Upstream Gradio demo using the already-loaded AI-Lab model |
| Higgs TTS 3 | Speak | Official SGLang-Omni playground connected to the already-loaded Higgs worker |
| VoxCPM2 | Speak | Upstream Gradio editor using the already-loaded model |
| Kokoro | Speak | Upstream Gradio editor using the already-loaded model |
| Khala | Music | Full Khala Studio frontend and its native queued Mac worker |
| YuE2 | Music | Full ds-yue-webui Studio: generate, cover, edit, ABC score and library |
| HeartMuLa | Music | Full HeartMuse Studio using the already-loaded pipeline |

Other engines remain available to API clients through AI-Lab's gateway, but
AI-Lab does not substitute small browser forms for their full interfaces.

MuLaCover has no verified full native editor. ASR, diarization, alignment,
OCR and Demucs/Matchering remain API/command utilities. These entries have no
direct-use button rather than a misleading one.
ACE-Step and Qwen3-TTS each expose their upstream Gradio editor on the
instance port plus 10000, in the same process as the API and with the same
model object; opening the editor does not load a second checkpoint. Their
Gradio queue is limited to one request at a time. API requests should not be
sent concurrently with a manual generation in that editor, since the two
interfaces do not share one request queue.

Higgs runs its official SGLang-Omni playground on the instance port plus
10000. This is a separate lightweight web process that forwards requests to
the existing Higgs worker; it does not load another checkpoint. Unloading
Higgs stops both the playground and the worker. Its browser interface allows
reference-audio uploads and generation controls that the AI-Lab gateway does
not expose.

VoxCPM2 uses its upstream Gradio editor on the instance port plus 10000.
Kokoro uses its upstream Gradio editor on the instance port plus 10000 and
keeps the official US/UK voice, speed, token, pronunciation and streaming
controls. It reuses the resident checkpoint; it does not load another model.
Khala starts its complete Studio stack with the AI-Lab instance: the upstream
React frontend, queue/progress dispatcher and the Apple-Silicon worker. The
worker owns the resident model and is terminated when the instance unloads.
YuE2 starts the pinned ds-yue-webui Studio and its resident YuE2 worker. Both
the Studio and the AI-Lab music endpoint submit work to that same worker, so
the checkpoint is not loaded twice.
HeartMuLa launches HeartMuse with composition fields, variants, style
reference, transcription, history and memory monitoring. HeartMuse receives
AI-Lab's existing pipeline and shares its generation lock, so UI and API work
are serialized around one checkpoint.

Each loaded ComfyUI instance has one supervised ComfyUI child process on its
AI-Lab port plus 10000. The link opens that process directly. Unloading the
instance stops the adapter and its child. Separate loaded instances have
separate ComfyUI processes and memory use. AI-Lab's named API workflow and
manual runs in native ComfyUI share that one process and its queue.

ComfyUI listens on the host's network interfaces so a browser on the private
network can reach it. This native interface has no additional authentication;
deployments must restrict access at the network/firewall layer. Do not expose
these ports to the public internet.

When a model-specific UI template is bundled, it is separate from the
API-format graph used by AI-Lab jobs. Unknown models fall back to the
configured API-format graph. Image,
audio or video input placeholders need to be filled in the native interface
before a manual run. The model checkpoint and any required components remain
configured in ComfyUI's per-instance model paths.
