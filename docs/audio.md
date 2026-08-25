# Audio inference

AI-Lab exposes the configured audio models through task-specific endpoints.
How callers obtain, store or otherwise prepare their source material is outside
this project's contract.

## How the evaluation audio was prepared

The Romanian evaluation audio published for this personal AI-Lab was prepared
by [Data-Lab](https://github.com/marianvid/data-lab). Its public repository has
one deliberately narrow purpose: it shows the deterministic FLEURS selection
and FFmpeg normalisation used before the files were sent here.

Data-Lab is not required to use these endpoints. Any caller may prepare input
according to the selected model's contract; the link records how this project's
published audio measurements were made.

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
`scipy`, `soundfile` and `soxr`). They are part of the inference runtime. The
separate Data-Lab preparation step described above is outside this runtime.

## Input contract

Transcription accepts the common audio files understood by the selected
engine. Silero is the one AI-Lab-specific exception worth documenting here: it
requires 16 kHz audio, converts stereo to mono, and refuses other sample rates
with a clear error. Other preparation follows the selected model and endpoint.

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
