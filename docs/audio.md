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
| transcription | `/v1/audio/transcriptions` | MLX Whisper (macOS only) | Whisper checkpoints in safetensors format, run on Apple silicon |
| voice activity detection | `/v1/audio/speech-segments` | ONNX Runtime adapter | Silero VAD 6.2.1 |
| alignment | `/v1/audio/alignments` | Qwen Forced Aligner (macOS only) | Qwen3 ForcedAligner; Romanian is not in its supported-language list |
| diarization | `/v1/audio/diarizations` | NeMo | NVIDIA Sortformer 4-speaker v1; non-commercial evaluation only |
| diarization | `/v1/audio/diarizations` | pyannote.audio | Speaker Diarization Community-1 |
| speech synthesis | `/v1/audio/speech/generations` | Qwen3-TTS, Kokoro, VoxCPM, Higgs TTS | see [Speech engines](#speech-engines) |

The task is a property of both a stored model and a configured instance. The
interface filters out engines that cannot serve that task instead of allowing a
configuration that will fail only when it starts.

## Which machine runs which audio engine

The Linux host and the Mac offer different engines (`hosts/linux.py` and
`hosts/darwin.py`). An engine that the host does not offer is not shown for
that host at all.

| Engine | What it does | Linux | macOS |
|---|---|---|---|
| vLLM | transcription (it also serves text models) | yes (NVIDIA card needed) | no |
| NVIDIA NeMo Speech | transcription and diarization from `.nemo` files | yes (NVIDIA card needed) | no |
| MLX Whisper | transcription on Apple silicon | no | yes |
| ONNX Runtime (Silero) | voice activity detection | yes | yes |
| pyannote.audio | diarization | yes | yes |
| Qwen Forced Aligner | alignment | no | yes |
| Qwen3-TTS | speech synthesis | yes | yes |
| Higgs TTS (SGLang-Omni) | speech synthesis | yes | no |
| Higgs TTS (transformers) | speech synthesis, same weights | no | yes |
| Kokoro | speech synthesis | no | yes |
| VoxCPM | speech synthesis | no | yes |

Diarization means working out who spoke when. Voice activity detection (VAD)
means finding the parts of a recording where somebody is speaking.

## Alignment

Alignment takes a recording and its known transcript, and says when each word
was spoken. Send a multipart request to `POST /v1/audio/alignments` with the
audio in `file`, the transcript in `text` (1–10000 characters) and the
`language` name (default `English`). The answer is
`{"words":[{"text":…,"start":…,"end":…}]}`, with times in seconds.

The aligner accepts Chinese, English, Cantonese, French, German, Italian,
Japanese, Korean, Portuguese, Russian and Spanish. Any other language is
refused, which is why it is not used for the Romanian evaluation audio.

## Speech engines

Speech synthesis (text to speech) goes to `POST /v1/audio/speech/generations`.
The request fields — `seed`, a reference voice to imitate, and which model
takes which — are described in
[Writing a request](requests.md#speech-a-seed-and-a-voice-to-imitate).

Higgs TTS comes in two forms that serve the same request contract. On Linux
it runs behind an SGLang-Omni worker, a separate serving program that AI-Lab
starts and watches over. The engine setting `max_parallel` (1–64) limits how
many requests that worker is sent at once. On the Mac it runs through the
transformers library inside AI-Lab's own adapter, because SGLang-Omni does not
run there. There, requests that arrive within `batch_window_ms` of each other
(default 100 ms) are grouped, up to `max_batch` (1–16), and decoded together.
Both forms offer the upstream Higgs playground page for direct use on the
instance port plus 10000 ([Use a loaded model directly](direct-use.md)).

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
work. It has one backend per engine: NeMo, MLX Whisper, Sortformer, pyannote,
Silero and the Qwen aligner. There is still one process per configured
instance, exactly as for text engines. On Linux systemd supervises it; on the
Mac the manager starts it as its own child process.

vLLM's base package does not include audio decoding. The active vLLM
environment therefore also carries its declared audio dependencies (`av`,
`scipy`, `soundfile` and `soxr`). They are part of the inference runtime. The
separate Data-Lab preparation step described above is outside this runtime.

## Input contract

Transcription accepts the common audio files understood by the selected
engine. The Silero adapter converts stereo to mono and resamples uploaded
audio to the 16 kHz input its model requires. Other preparation follows the
selected model and endpoint.

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
