# Model storage layout

This document describes how model weights are organised on the AI-Lab host and
why. It is written to be useful later, when engines other than llama.cpp are
added — the directory layout is deliberately chosen so that adding vLLM,
SGLang or TensorRT-LLM does not require moving anything.

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

`downloads/` is the only directory owned by `ai-lab-manager`; the rest are owned
by root inside the container. On the host these appear as uid `100999`/`100000`
respectively, because LXC `102` is unprivileged and shifts uids by 100000.

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

## Adding a second engine later

Nothing here is llama.cpp specific except the contents of `gguf/`. A future
vLLM instance would:

1. download into `fp8/` or `nvfp4/` with `HF_HOME=/models/cache/hf`;
2. register that directory as a storage location in `config.json`;
3. run under the same `ai-lab-engine@` unit as everything else.

Nothing in the application assumes GGUF any more. `ai_lab/engines/vllm.py`
already declares which formats it reads; what is missing is the command line it
would build and the settings it would accept. Instances already carry an
explicit `engine` field, and the templated systemd unit serves any engine, so
no plumbing has to change.

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
