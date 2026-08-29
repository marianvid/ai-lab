# Model storage layout

## Core and benchmark tiers

AI-Lab uses two explicit storage tiers. "core" is the production library on
the internal Lexar. "benchmark" is disposable test capacity on the external
Corsair. A missing benchmark disk never redirects a download to core.

| Tier | Host | LXC | Purpose |
|---|---|---|---|
| core | /mnt/ai-models | /models | production models |
| benchmark | /mnt/corsair-4tb/test_models | /test_models | downloads and benchmarks |

Library can promote or demote a model. AI-Lab copies into a temporary path,
verifies every file by SHA-256, commits the destination, and only then removes
the source. Models referenced by an instance cannot move.

This document describes how model weights are organised on the AI-Lab host and
why. The layout was chosen before multiple engines arrived, and now lets
llama.cpp, vLLM, NeMo and ONNX-backed audio services coexist without moving or
duplicating weights. It leaves room for SGLang or TensorRT-LLM later.

## Where the weights live

| Layer | Path |
|---|---|
| Physical disk | Lexar NM790 4 TB, the internal one that also holds the system |
| Volume | `pve/ai-models`, a thin volume of 800 GB, ext4 label `ai-models` |
| Host mount | `/mnt/ai-models` (from `/etc/fstab`, `nofail`) |
| Path inside LXC `102` | `/models` (bind mount declared as `mp0` in `102.conf`) |

The container sees only `/models`. Nothing in `config.json` or in the systemd
units refers to the host path, so the store can be moved to a different disk by
changing one line in `/etc/pve/lxc/102.conf` and restarting the container.

### It used to be the external disk, and that is why it moved

Until 19 August 2026 the store was the **Corsair EX400U**, an external NVMe on a
USB4 link, mounted at `/mnt/corsair-4tb`. That disk kept falling off the
Thunderbolt bus — nine times on 19 August alone, at one point every few minutes —
and each time it took AI-Lab with it: every read of `/models` returned
`Input/output error` and no engine could start.

The 447 GB were copied to the internal disk, which had 3.5 TB free in the thin
pool and is **faster anyway**: 4.0/4.5 GB/s against 3.3/3.0 GB/s through USB4.
The copy ran at 560 MB/s and verified equal, 471 files on both sides.

**AI-Lab no longer depends on any external disk.** That is the point of the
move, more than the speed: the store is now on the same disk as everything else
that has to be working for the container to run at all.

The external disk is still attached and still labelled `corsair-4tb`. It holds
the media library work and whatever the home-lab dashboard reads. It is no
longer AI-Lab's problem.

## Directory layout

The first level is the **weight format**, not the engine or the model family.
Formats outlive engines: vLLM, SGLang and TensorRT-LLM all consume safetensors
variants, while GGUF is consumed by llama.cpp and increasingly by others. A
format-first tree also allows mounting individual subtrees read-only later.

```
/models/
├── audio/           ASR, alignment, VAD and diarization model repositories
│   ├── asr/
│   ├── alignment/
│   ├── vad/
│   └── diarization/
├── gguf/            llama.cpp, ollama — quantised single files
│   ├── qwen-coder/
│   └── gemma-general/
├── safetensors/     unquantised HF repos (BF16/FP16), the source for derived quants
├── fp8/             FP8 quantised safetensors — Blackwell-native
├── nvfp4/           NVFP4/MXFP4 quantised safetensors — Blackwell-native
├── awq/             AWQ 4-bit — legacy from the Ampere era, kept for compatibility
├── gptq/            GPTQ 4/8-bit — same
├── exl/             EXL2/EXL3 for exllamav2/v3, if ever needed
├── engines/
│   └── trtllm/      compiled TensorRT-LLM engines, per GPU + driver + TRT version
├── downloads/       staging area and partial downloads (writable by the manager)
└── cache/
    └── hf/          HF_HOME — keeps huggingface_hub off the container root disk
```

Audio models keep their upstream runtime layout. Hugging Face ASR and alignment
repositories remain whole directories, NVIDIA NeMo checkpoints remain native
`.nemo` files, and VAD packages retain the code and weights shipped together by
their upstream project. They are served only by AI-Lab; data-processing clients
never mount this tree or load the weights directly.

`downloads/`, `cache/hf/` and `audio/` are owned by `ai-lab-manager`; the stable
language-model trees are owned by root inside the container. On the host these
appear as shifted LXC uids because container `102` is unprivileged.

### Two directories that are easy to forget

`downloads/` and `cache/hf/` exist because both `huggingface_hub` and vLLM
default to writing into `~/.cache/huggingface`, which lives on the container's
512 GB root disk. Setting `HF_HOME=/models/cache/hf` before running any
downloader keeps that traffic on the model volume instead.

`engines/trtllm/` holds build artefacts, not models. They are tied to a specific
GPU, driver and TensorRT version, and are meant to be regenerated rather than
preserved — treat that subtree as disposable.

### What counts as one model

GGUF and everything else are counted differently, and AI-Lab follows the same
rule:

- **GGUF: one file is one model.** Two `.gguf` files in a directory are two
  models. A model split into `-00001-of-00003` parts is still one, because the
  shard numbering says so.
- **Every other format: one directory is one model.** The weights are spread over
  several `.safetensors` files plus a tokenizer and a `config.json`, and the
  engine is handed the whole directory, never a single file.

Two consequences that caused real bugs:

- **Do not name a model after its first weight file.** Every safetensors model's
  first file is called `model-00001-of-....safetensors`, so naming that way gives
  every model the same name, `model`. The directory name is the model name.
- **Some weight files are only a part.** `model_mtp.safetensors` (a small
  extra-prediction head) and `model-towers.safetensors` sit beside the main
  weights and cannot be loaded on their own. They are components of the model in
  that directory, not models.

So a directory holding five weight files plus a tokenizer is **one** thing to
show in the interface, not six.

### An image model is several files, and still one model

The directory-is-the-model rule does the work here too, but the files arrive
from further apart. A ComfyUI image model is assembled from parts that live in
different folders of a repository, and sometimes in different repositories:
the diffusion model, the text encoder that reads the prompt, and the VAE that
turns the result into a picture.

They are stored **flat, in one directory**, under their own upstream file
names:

```
/test_models/images/generation/qwen-image-2512-nvfp4/
├── qwen_image_nvfp4.safetensors          the diffusion model
├── qwen_2.5_vl_7b_nvfp4.safetensors      the text encoder
└── qwen_image_vae.safetensors            the VAE
```

The folders they sat in upstream are not recreated, because ComfyUI is pointed
at this one directory for every category it looks a file up under, and it finds
each part by name. Which parts belong together is declared in configuration
rather than guessed; see [the Library](docs/library.md#models-that-are-made-of-parts).

A part used by two models is stored once per model. The alternative — one copy
with links to it — makes deleting either model a question about the other, and
the disk this is kept on has room.

### A projector is weights, but it is not a model

`mmproj-gemma-4-26B-A4B-it-f16.gguf` sits beside a GGUF model and holds the part
that turns a picture into something the model can read. It ends in `.gguf` and it
really is weights, but llama.cpp is handed it with `--mmproj` *alongside* the
model; started on its own it serves nothing.

It is the GGUF version of the `model_mtp.safetensors` problem above. For every
other format the directory-is-the-model rule takes care of it. GGUF has no such
rule — there one file genuinely is one model — so this one is named explicitly,
in `naming.py`. A projector is attached to the model in its directory, the way a
tokenizer is, and never listed on its own.

Left as a model it becomes a library entry that can never be started, and an
entry pointing at it blocks deleting the real model beside it.

## The same layout on the Mac

The M3 Max keeps its models at `/Volumes/Marian_Backup/models`, with the same
format-first tree — `gguf/`, `safetensors/`, `fp8/`, `nvfp4/` — so a path means
the same thing on both machines and a model can be moved across without being
reorganised.

**The application now enforces this.** One directory is configured — the models
root — and each repository is a folder in it named after its weight format.
Setting them one at a time let GGUF sit on one disk and NVFP4 on another, which
nothing else here expects and which nobody chooses on purpose. A configuration
written before that still loads: the root is read back from the directory the
repositories shared, so `/models` on the container and
`/Volumes/Marian_Backup/models` on the Mac come out unchanged.

The *format* names the folder, not the repository's id — an id is a short name
somebody chose and may differ.

Two differences that are not going away:

- **Only `gguf/` has anything in it, and only llama.cpp reads it.** vLLM needs
  CUDA; Apple silicon offers Metal. The interface shows vLLM greyed out with
  that reason rather than pretending it might work. `fp8/` and `nvfp4/` exist
  for symmetry and are empty.
- **Memory is unified**, so there is no separate pool of video memory to wait
  for after an unload. The step that polls until the card goes quiet is skipped
  entirely, and the readings report the engine process's resident memory
  instead.

Weights arrive through the Hugging Face cache and do not stay there. That cache
keeps a repository directory of hashed blobs with a snapshot of symlinks over
them, which is a fine way to download and a poor way to keep anything: the name
on disk is a hash, one file can be a snapshot of another, and nothing says which
quantisation you are looking at. Moving a model into the store is what makes it
a model rather than a download.

## Weight formats on NVIDIA hardware

Relevant context for choosing what to download, given a single RTX PRO 4500
Blackwell with 32 GB of VRAM.

| Format | Consumed by | Notes |
|---|---|---|
| GGUF | llama.cpp, ollama | One quantised file per model. What is in use today. |
| Safetensors (BF16/FP16) | vLLM, SGLang, transformers | A directory, not a file. The unquantised source. |
| FP8 | vLLM, SGLang, TensorRT-LLM | Native tensor-core support on Blackwell. |
| NVFP4 / MXFP4 | vLLM, TensorRT-LLM | Native 4-bit on Blackwell; produced by TensorRT Model Optimizer or llm-compressor. |
| AWQ, GPTQ | vLLM, SGLang | Pre-Blackwell 4-bit schemes. Work, but leave performance unused. |
| `compressed-tensors` | vLLM | Neural Magic / Red Hat container format wrapping several schemes. |
| TensorRT-LLM engine | TensorRT-LLM only | Compiled, non-portable, rebuilt per GPU and version. |

The VRAM budget is the binding constraint. A 35B model in BF16 needs roughly
70 GB and does not fit; anything served through vLLM on this machine will be
quantised. Since Blackwell has native FP8 and FP4 tensor cores, **FP8 and NVFP4
are the formats worth targeting** — AWQ and GPTQ would run but waste the
hardware's capability.

## Multiple engines now share the store

Nothing here is llama.cpp specific except the contents of `gguf/`. vLLM reads
the native and quantised safetensors trees, NeMo restores native `.nemo`
checkpoints under `audio/asr/`, and the ONNX adapter serves Silero from
`audio/vad/`. All run under the same templated systemd supervision.

A repository can now declare both a task and a subtree. That is how two catalog
views share `audio/asr/`: the safetensors transcription repository recognises
Hugging Face directories, while the NeMo repository recognises `.nemo`
checkpoints. Neither lists the other's format and no model is duplicated.

## Performance note

Measured with direct I/O on 8 GB transfers:

| Disk | Read | Write |
|---|---:|---:|
| Corsair EX400U (USB4) | 3.3 GB/s | 3.0 GB/s |
| Lexar NM790 (internal PCIe) | 4.0 GB/s | 4.5 GB/s |

Storage bandwidth only affects model load time, never inference throughput, so
the external disk costs a little over a second on a 25 GB load. The reported
PCIe link speed of `2.5GT/s x1` is an artefact of USB4 tunnelling and does not
reflect real throughput.

`lexar-2` is reserved for real-time data capture and is not available for model
storage, regardless of how empty it looks.

## What is actually stored, 17 August 2026

`nvfp4/` and a second `gguf/` entry stopped being theoretical on 17 August 2026,
when vLLM was installed and the first native-format models were downloaded.

| Path | Model | Size | Engine |
|---|---|---:|---|
| `gguf/qwen-coder/` | Qwen3.6-35B-A3B UD-Q4_K_M | 22 GB | llama.cpp |
| `gguf/gemma-general/` | Gemma-4-E4B-it Q4_0 | 4.6 GB | llama.cpp |
| `gguf/qwen3-coder-next/` | Qwen3-Coder-Next 80B-A3B UD-Q4_K_XL | 47 GB | llama.cpp |
| `nvfp4/qwen3-coder-30b-a3b/` | Qwen3-Coder-30B-A3B-Instruct | 17 GB | vLLM |
| `nvfp4/qwopus3.6-27b-coder/` | Qwopus3.6-27B-Coder | 20 GB | vLLM |

The format-first layout held up: adding vLLM required no file to move.

## What is actually stored, 25 August 2026 — audio

| Path | Model | Weights | Engine | Task |
|---|---|---:|---|---|
| `audio/asr/whisper-large-v3/` | Whisper large-v3 | 3.09 GB | vLLM | transcription |
| `audio/asr/whisper-large-v3-turbo/` | Whisper large-v3-turbo | 1.62 GB | vLLM | transcription |
| `audio/asr/qwen3-asr-0.6b/` | Qwen3-ASR-0.6B | 1.88 GB | vLLM | transcription |
| `audio/asr/qwen3-asr-1.7b/` | Qwen3-ASR-1.7B | 4.70 GB | vLLM | transcription |
| `audio/asr/parakeet-tdt-0.6b-v3/` | Parakeet TDT 0.6B v3 | 2.51 GB | NeMo | transcription |
| `audio/asr/canary-1b-v2/` | Canary 1B v2 | 6.36 GB | NeMo | transcription |
| `audio/asr/nemotron-3.5-asr-streaming-0.6b/` | Nemotron 3.5 ASR Streaming 0.6B | 2.37 GB | NeMo | transcription |
| `audio/alignment/qwen3-forced-aligner-0.6b/` | Qwen3 ForcedAligner 0.6B | 1.84 GB | not configured for Romanian | alignment |
| `audio/vad/silero-vad/` | Silero VAD 6.2.1 | package repository | ONNX Runtime | VAD |

The forced aligner stays on disk for possible international use, but Romanian
is not among the languages its model card supports. The diarization directory
is intentionally empty: the commercial-compatible Pyannote checkpoint is
gated, while the inspected NVIDIA Sortformer checkpoint is non-commercial.

### The size limit is the real limit

A 32 GB card cannot run a model whose smallest native checkpoint is larger than
32 GB, however well the silicon supports the format. Qwen3-Coder-Next is the
clear case — its FP8 checkpoint is 80.4 GB and its NVFP4 checkpoint 47.6 GB, so
neither fits, and the model can only be run as GGUF with its experts pushed into
system RAM.

NVFP4 is not smaller than a 4-bit GGUF. Both spend about four bits per
parameter. What NVFP4 buys is speed, and mostly the speed of *reading* a prompt
rather than of writing an answer — see the measurements in
`opts/HOMELAB_ENVIRONMENT.md`, in the private half.

Practical ceiling for anything that must live entirely in VRAM: roughly **23 GB
of weights**, leaving room for the context cache.
