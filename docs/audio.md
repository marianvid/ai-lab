# Audio — inference belongs here, data handling does not

AI-Lab owns models and inference. A data-processing client owns downloads,
FFmpeg conversion, long-file chunking, durable storage, orchestration and
metrics. The client sends normalised audio over the private network and gets
model output back; it never mounts `/models` and never installs an AI runtime.

That boundary matters when one client becomes many. Public-meeting processing,
interview analysis and archive indexing can share the expensive, versioned
inference layer without sharing source data or workflow-specific code.

## Tasks and engines

| Task | Endpoint | Current engine | Current models |
|---|---|---|---|
| transcription | `/v1/audio/transcriptions` | vLLM | Whisper large-v3, Whisper large-v3-turbo, Qwen3-ASR 0.6B and 1.7B |
| transcription | `/v1/audio/transcriptions` | NeMo | Parakeet TDT 0.6B v3, Canary 1B v2; Nemotron 3.5 ASR Streaming 0.6B is installed but its checkpoint does not accept Romanian |
| voice activity detection | `/v1/audio/speech-segments` | ONNX Runtime adapter | Silero VAD 6.2.1 |
| alignment | — | not served for Romanian | Qwen3 ForcedAligner is stored, but Romanian is not in its supported-language list |
| diarization | `/v1/audio/diarizations` | NeMo | NVIDIA Sortformer 4-speaker v1; non-commercial evaluation only |
| diarization | `/v1/audio/diarizations` | pyannote.audio | Speaker Diarization Community-1 |

The task is a property of both a stored model and a configured instance. The
interface filters out engines that cannot serve that task instead of allowing a
configuration that will fail only when it starts.

## Runtime isolation

The manager, vLLM, NeMo and Silero do not share Python packages:

```text
/opt/ai-lab                 manager and adapters
/opt/ai/vllm/current       vLLM environment selected by a stable link
/opt/ai/nemo/current       selected NeMo environment and CUDA dependencies
/opt/ai/silero/current     selected Silero, CPU Torch and ONNX environment
/opt/ai/pyannote/current   selected pyannote.audio and CUDA environment
```

The adapter in `ai_lab/audio/server.py` is launched with the runtime's Python.
It restores one model, exposes a small HTTP contract, and does no durable data
work. systemd still supervises one process per configured instance, exactly as
it does for text engines.

vLLM's base package does not include audio decoding. The active vLLM
environment therefore also carries its declared audio dependencies (`av`,
`scipy`, `soundfile` and `soxr`). They are part of the inference runtime, not a
reason to move decoding or source-file preparation out of the client.

## Input contract

Transcription accepts common audio files understood by the selected engine.
For reproducible bulk work the client should produce mono 16 kHz PCM WAV before
sending it. Silero currently requires 16 kHz and converts stereo to mono inside
the inference boundary; other sample rates are refused with a clear error.

Very long recordings should be downloaded once by the data client, converted
once, split at technical or VAD boundaries, and then sent in pieces. This avoids
moving a whole video through AI-Lab and allows retrying only the failed piece.

## Licensing and evaluation

The configured collection includes one NC checkpoint, Sortformer 4-speaker
v1, strictly for personal comparative evaluation. It is clearly marked and is
not a production candidate. Pyannote Community-1 is CC BY 4.0.
The exact upstream repository, revision and licence of every audio model are
kept with the private deployment inventory. Public, reproducible Romanian
measurements live in
[ai-lab-benchmarks](https://github.com/marianvid/ai-lab-benchmarks); evaluation
audio is fetched from its publisher and is not committed. The Echo dataset
card does not currently state a licence; its benchmark audio therefore remains
only in the private evaluation store and is not redistributed.

---

[← all documents](../README.md)  ·  [Writing a request](requests.md)  ·  [Models](models.md)
