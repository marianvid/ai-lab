# Model support is the constraint, not speed

> **Corrected later.** Part of this document rejects TensorRT-LLM and SGLang
> using throughput numbers published from other hardware — the exact mistake
> `04-PITFALLS.md` warns against. Both were afterwards run here, on this card.
> The conclusion survived, but two of the published claims did not.
> See `10-FOUR-ENGINES-MEASURED.md`.

A day spent trying to add TensorRT-LLM as a third engine produced a finding
worth more than the benchmark that was intended.

**Every inference engine needs code written by hand for each model
architecture.** None of them can load an architecture they have never seen. What
separates them is not whether they need that code, but how fast it appears.

## Measured on this machine, 18 August 2026, with Gemma-4

| Engine | Gemma-4 support |
|---|---|
| **llama.cpp** build 10449 | works; measured on four tests |
| **vLLM** 0.27.1 stable | code present but **broken** — two defects, patched locally |
| **TensorRT-LLM** 1.2.1 | **no code at all** |

The C++ project maintained largely by volunteers has the best coverage. The
GPU vendor's own optimised engine has the worst.

## Why TensorRT-LLM cannot run Gemma-4

Not a bug. Three independent walls, any one of which is fatal:

1. **No implementation.** `_torch/models/` contains `modeling_gemma3.py`. There
   is no `gemma4`.
2. **The pinned library predates the model.** The package requires
   `transformers==4.57.3` exactly. That version's config registry knows `gemma`,
   `gemma2`, `gemma3`, `gemma3n` — and stops. Gemma-4 was released in April 2026.
3. **The documentation describes a different tree.** NVIDIA's supported-models
   page lists `Gemma4ForConditionalGeneration` and
   `Gemma4UnifiedForConditionalGeneration`. That page is built from `main`, not
   from the release. Reading it and installing the release gives two different
   products.

Three of the four models on this machine are too new for the pinned library:

| Model | `model_type` | In transformers 4.57.3 |
|---|---|---|
| gemma-4-26b-a4b | `gemma4` | no |
| glm-4.7-flash | `glm4_moe_lite` | no |
| qwopus3.6-27b-coder | `qwen3_5` | no |
| qwen3-coder-30b-a3b | `qwen3_moe` | yes |

## Going back to an older version makes it worse

The intuition was reasonable — older TensorRT-LLM still has the compiled-engine
path, which is what the speed argument rested on. But an older release pins an
older `transformers`, so the catalogue shrinks further. You would buy
ahead-of-time compilation and pay for it with even fewer runnable models.

## The compiled-engine path no longer exists anyway

Release 1.2 removed the TensorRT backend entirely: `trtllm-build`,
`trtllm-refit`, `trtllm-prune`, the per-model `convert_checkpoint.py` scripts,
and the `tensorrt` dependency. The documentation states *"PyTorch is now the sole
execution backend"* and *"there is no separate checkpoint-conversion or
engine-build step."*

**NVIDIA's stated reason is model support velocity** — consolidating on one
backend so new architectures reach functional support faster, citing MiniMax M3
supported within a week of release.

So the removal and the coverage problem have the same cause. Per-model
ahead-of-time compilation could not keep up with the release cadence, and NVIDIA
chose breadth over peak performance. The loss is acknowledged: an open issue
reports *"performance degradation using PyTorch backend compared to TensorRT
backend"*.

Everything previously written here about `max_input_len`, `max_batch_size` and a
fixed shape envelope describes a workflow that has been deleted. The article
length measurements survive as useful data about the corpus; their original
purpose does not.

## What this means for a three-engine plan

The plan was: TensorRT-LLM as the production workhorse, vLLM and llama.cpp for
flexibility. The logic holds. **The calendar breaks it.**

By the time TensorRT-LLM supports a model, it has been running on vLLM for
months. Switching would mean waiting for a model to become old enough to be
supported — and by then there is usually a better one.

This does not disqualify TensorRT-LLM generally. Anyone running Llama or Mistral
in production for a year gains real throughput. It disqualifies it for someone
testing new models weekly.

## The gap is stable-versus-nightly, and it cuts both ways

vLLM's own recipe for Gemma-4 requires the **nightly** build:

```
uv pip install -U vllm --pre \
  --extra-index-url https://wheels.vllm.ai/nightly/cu129 \
  --extra-index-url https://download.pytorch.org/whl/cu129 \
  --index-strategy unsafe-best-match
```

So the local patch in `05-VLLM-GEMMA4-BUG.md` is not a discovery of an unknown
defect. It reproduces on the stable release what already exists upstream. That is
a better thing to publish than a patch: *stable is broken, nightly works, here is
the patch if you are pinned to stable.*

The same applies to TensorRT-LLM: NVIDIA publishes development builds at
`https://pypi.nvidia.com`, and the documentation that lists Gemma-4 describes
that tree.

## The development build was tried too, and hit a different wall

Since the released version is too old for the models, the obvious next step was
NVIDIA's development channel at `https://pypi.nvidia.com`. That build **does**
have what the documentation promised:

- `tensorrt_llm 1.3.0rc24`, with `transformers 5.5.4`
- `gemma4` present in the transformers config registry
- `_torch/models/modeling_gemma4.py`, `modeling_gemma4_unified.py`,
  `modeling_gemma4_vision.py`, `modeling_gemma4_audio.py`, `modeling_gemma4mm.py`

And then it failed to compile its kernels:

```
flashinfer/trtllm/fmha/fmhaKernels.cuh(136): error: identifier
  "CU_FUNC_ATTRIBUTE_SHARED_MEMORY_MODE" is undefined
  "CU_SHARED_MEMORY_MODE_ALLOW_OVERSIZED_SHARED_MEMORY" is undefined
  "CU_DEVICE_ATTRIBUTE_MAX_OVERSIZED_SHARED_MEMORY_PER_BLOCK" is undefined
```

Those symbols are not in CUDA 13.3's `cuda.h`. Two ways out were tried:

1. **A different attention backend** (`attn_backend: FLASHINFER` instead of
   `TRTLLM`). No help — the failing file is `trtllm_fmha_kernel_launcher.cu`
   inside FlashInfer, which is compiled regardless of the backend chosen.
2. **An older FlashInfer** (0.6.15). It compiled, and then TensorRT-LLM failed
   on its own API expectations:
   `'TrtllmAttentionMetadata' object has no attribute
   'swap_paged_kv_indices_for_layer'`. One incompatibility traded for another.

## The CUDA ceiling is not this machine being behind

Checked rather than assumed:

| | |
|---|---|
| Driver installed | **610.57.04** |
| Newest driver in NVIDIA's `debian13` repo | **610.57.04** — already the newest |
| "Latest production" in NVIDIA's March 2026 compatibility matrix | 595 |
| CUDA reported by the driver | 13.3 |
| Newest published CUDA toolkit | 13.3 |

This machine runs the newest published driver, roughly fifteen versions ahead of
what NVIDIA labels *production*, and the newest published toolkit.

**CUDA 13.4 is a Developer Preview, not a release.** The symbols FlashInfer wants
come from it. So `1.3.0rc24` targets software NVIDIA has not shipped.

The framing matters: this is not "the CUDA here is too old". It is "the
development build requires an unreleased CUDA".

## Would another distribution help? Measured: no

| NVIDIA repo | Newest CUDA toolkit |
|---|---|
| `debian13` | 13-3 |
| `ubuntu2404` | 13-3 |
| `ubuntu2604` | 13-3 |

Identical ceilings. Ubuntu carries more *old* versions because it has been
supported longer; the top is the same. The 13.4 preview is in none of them.

Three further points settled while looking:

- **An LXC container uses the host's driver.** It shares the kernel; there is no
  separate driver inside. Reinstalling the container with Ubuntu would change
  userspace and nothing about CUDA.
- **Proxmox VE exists only on Debian.** There is no Ubuntu build, official or
  otherwise. Its kernel is derived from Ubuntu's, so the distance from Ubuntu is
  smaller than the distribution name suggests.
- **The real architectural lever is container versus virtual machine.** With PCIe
  passthrough, a VM owns the card and carries its own driver, so a preview driver
  could be tested there without touching the host. Proxmox does passthrough as
  well as anything else, so this is not a reason to change platform. The costs
  are losing GPU sharing between containers, and OCuLink passthrough being
  awkward on a link that does not support hot-plug.

Ubuntu's only real edge is **timing**: NVIDIA sometimes publishes there first,
and most documentation assumes it. Neither is a capability difference.

## Eight walls, in order

For the record, because the pattern is the finding:

| # | Wall |
|---|---|
| 1 | released 1.2.1 has no `modeling_gemma4.py` |
| 2 | it pins `transformers==4.57.3`, which predates Gemma 4, GLM-4.7 and Qwen3.5 |
| 3 | MPI worker spawn fails in an unprivileged LXC (`MPI_ERR_UNKNOWN`) |
| 4 | `TLLM_DISABLE_MPI=1` needs Ray, absent |
| 5 | Ray 2.57 is incompatible: `PlacementGroupSchedulingStrategy` import fails |
| 6 | dev build has a broken `sympy`/`mpmath` pair — fixed with `mpmath<1.4` |
| 7 | its FlashInfer kernels need CUDA symbols absent from 13.3 |
| 8 | downgrading FlashInfer breaks TensorRT-LLM's own API expectations |

Meanwhile, on the same machine, the same day: **llama.cpp runs Gemma-4 without
complaint, and so does vLLM nightly.**

## And vLLM nightly needs no patch at all

The local fix in `05-VLLM-GEMMA4-BUG.md` was tested against the alternative.
`config.json` was restored to its original state, and vLLM nightly
(`0.27.2rc1.dev205`, transformers 5.15) was installed in a third environment and
pointed at the unmodified model.

It loaded. Zero occurrences of either error the patch addresses.

| | nightly, unpatched | stable 0.27.1 + patch |
|---|---:|---:|
| Classification F1 | 0.9710 | 0.9693 |
| Translation chrF++ | **69.79** | 69.68 |
| Articles/s at c=32 | 158.2 | 159.6 |

Equivalent on every measurement. **The patch is only worth carrying if you are
pinned to a stable release.**

One caution, met for the third time: the first run after the new install
measured 108 articles/s, the second 116, the third 158. The kernel autotune cache
had to settle. A single measurement would have reported a 30% regression that
does not exist.

## The rule to take away

**Check that the engine's released version supports the model before installing
anything.** Two hours were lost twice in one day to documentation that described
a branch rather than a release. The check is cheap:

```python
# does the pinned transformers even know this architecture?
from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
print("gemma4" in CONFIG_MAPPING_NAMES)

# does the engine have model code for it?
from vllm.model_executor.models.registry import ModelRegistry
print([a for a in ModelRegistry.get_supported_archs() if "Gemma4" in a])
```

Both answers take seconds and would have saved most of a day.

## Also learned along the way

- TensorRT-LLM spawns MPI worker processes, which **an unprivileged LXC container
  cannot do** — `MPI_ERR_UNKNOWN` on `COMM_SELF.Spawn`. Basic MPI works; dynamic
  spawning does not. `TLLM_DISABLE_MPI=1` switches to Ray, which must then be
  installed and version-matched.
- It needs `libopenmpi-dev` and `openmpi-bin` present, or `mpi4py` cannot even
  import.
- vLLM's Gemma-4 recipe recommends two flags never tested here:
  `--async-scheduling` for throughput, and
  `--limit-mm-per-prompt image=0,audio=0` as an alternative to
  `--language-model-only`.

---

# The search for a C++ alternative, and how it ended

The question was reasonable: a C++ binary runs closer to the kernel, so it ought
to be faster than a Python server. Every candidate was checked before installing
anything — a rule adopted after losing hours twice to documentation that
described a branch rather than a release.

## MLC-LLM — compiled, native, and months behind

TVM compiles the model to native code for the target, with a C++ runtime, an
OpenAI-compatible server, continuous batching and paged KV cache. On paper the
closest thing to "vLLM in C++".

**It does not support Gemma-4.** Issue #3477 on `mlc-ai/mlc-llm`, opened
5 April 2026, still open in August: no assignee, no pull request. Trying it gives
`ValueError: Unknown model type: gemma4`.

The obstacles listed in that issue are the same three met here: weights nested
under `model.language_model.*`, the existing Gemma loader not knowing the vision
and audio towers, and fields like `query_pre_attn_scalar` living only in
`text_config`. vLLM solved them in nightly; MLC has not touched them in four
months.

No evidence was found of NVFP4 support on Blackwell either. The documented
`sm_120` NVFP4 ecosystem is vLLM, TensorRT-LLM and llm-compressor.

## LMDeploy TurboMind — genuinely C++, wrong formats

TurboMind is *"implemented almost entirely in C++ and CUDA"*, with persistent
batching. This is the real article, not a compromise.

Its architecture registry does contain two models already on this machine:

```
Qwen3_5ForConditionalGeneration = 'qwen3_5'        <- Qwopus3.6-27B-Coder
Glm4MoeLiteForCausalLM          = 'glm4-moe-lite'  <- GLM-4.7-Flash
Qwen3MoeForCausalLM             = 'qwen3-moe'
GptOssForCausalLM               = 'gpt-oss'
```

Gemma-3 and Gemma-4 are absent entirely.

**But the quantisation support settles it.** TurboMind handles FP16/BF16, W4A16
(INT4 via AWQ or GPTQ) and INT8/INT4 KV cache. **No FP8. No NVFP4.**

Running those models would mean re-quantising to INT4 AWQ — a pre-Blackwell
scheme. `MODEL_STORAGE.md` in this repository already described what that costs:
*"pre-Blackwell 4-bit schemes. Work, but leave performance unused."* You would
buy the C++ engine with the format the card was chosen for.

## SGLang — supported, but the evidence points the other way

SGLang does support Gemma-4 and added NVFP4 in its August 2026 release,
including NVFP4 KV cache for `sm_120`. It is a serious engine, not a curiosity.

A published head-to-head on Gemma-4 reports **vLLM ahead by 3× on time to first
token and 3× on concurrent throughput**. That was a BF16 baseline rather than
NVFP4, so it is not conclusive, but nothing suggests a win.

One detail worth carrying: **workstation Blackwell reports `sm_120` and does not
support FlashAttention-4.** The newest attention optimisations target B200, not
this class of card.

## Where it lands

| Engine | C++ | Gemma-4 | NVFP4 | Batching gain |
|---|---|---|---|---|
| **vLLM** | no | yes | yes | **×17.6 measured** |
| SGLang | no | yes | yes | reported 3× below vLLM |
| **llama.cpp** | **yes** | yes | no — GGUF only | ×1.4 measured |
| LMDeploy TurboMind | **yes** | no | **no** | good, untested here |
| TensorRT-LLM | no, PyTorch since 1.2 | dev build only | yes | never started |
| MLC-LLM | yes, via TVM | **no** | undocumented | — |

**No C++ engine supports NVFP4.** The native format this card was bought for is
served only by Python engines.

## Why the C++ intuition does not pay here

It was right in principle and irrelevant in practice, for two measured reasons.

**CUDA graphs remove the CPU from the inner loop.** Disabling them cost 73% of
throughput at one request and 56% at thirty-two — that is the size of the
CPU-side submission cost, and graphs eliminate it. Once the graph is captured the
CPU issues one replay per step, and whether Python or C++ issues it does not
matter.

**Python's cost is startup, not steady state.** Measured: 5 seconds of imports
and torch initialisation before vLLM writes its first log line, out of a 45-second
start. Real, but paid once.

## The conclusion

Two engines, two roles, both already installed and measured:

- **vLLM** when throughput matters — bulk classification, anything with
  concurrency. Ten to seventeen times the throughput of the alternative.
- **llama.cpp** when startup and model coverage matter — swapping models often,
  or running something too new for anyone else.

There is no third engine worth adding. The search is closed.
