# Results

All numbers measured on the RTX PRO 4500 Blackwell, 17–18 August 2026.
Raw files in `raw-results/`; consolidated in `data/dataset.json`.

Read `01-METHODOLOGY.md` for what the columns mean, and `04-PITFALLS.md` before
trusting anything.

## The models

| Model | Platform | On disk | VRAM used | Notes |
|---|---|---:|---:|---|
| Qwen3-Coder-30B-A3B | vLLM NVFP4 | 17 GB | 29 740 | MoE, 3B active |
| Qwopus3.6-27B-Coder | vLLM NVFP4 | 20.6 GB | 27 382 | dense, reasoning model |
| Gemma-4-26B-A4B | vLLM NVFP4 | 18.8 GB | 28 842 | MoE; needs the patch |
| GLM-4.7-Flash | vLLM NVFP4 | 21.3 GB | 28 350 | |
| Gemma-4-26B-A4B | GGUF Q4_K_XL | 17 GB | 18 184 | same model, other format |
| Gemma-4-31B | GGUF Q4_K_XL | 18.8 GB | 24 614 | dense |
| Qwen3.6-35B-A3B | GGUF Q4_K_M | 22 GB | 21 880 | **in production today** |
| Gemma-4-E4B | GGUF Q4_0 | 4.6 GB | 3 902 | **in production today** |
| Coder-Next REAP-40B | GGUF Q4_K_XL | 28.5 GB | 28 160 | pruned from 80B |
| Coder-Next REAP-48B | GGUF Q4_K_XL | 33.4 GB | 30 834 | 4 layers in RAM |
| Coder-Next REAP-60B | GGUF Q4_K_XL | 40.9 GB | 30 704 | 14 layers in RAM |
| Coder-Next 80B | GGUF Q4_K_XL | 47 GB | 30 562 | 20 layers in RAM |

## Section 1 — Coding

Ten Python tasks, executed against hidden tests. Claude Opus 5 reference
solutions score **10/10**; they are in `benchmark/reference-solutions/`.

| Model | Platform | Score | Failed on |
|---|---|---:|---|
| Coder-Next 80B | GGUF | **10/10** | — |
| Coder-Next REAP-60B | GGUF | **10/10** | — |
| Coder-Next REAP-48B | GGUF | **10/10** | — |
| Coder-Next REAP-40B | GGUF | **10/10** | — |
| Gemma-4-31B | GGUF | **10/10** | — |
| Gemma-4-26B-A4B | GGUF | **10/10** | — |
| Gemma-4-26B-A4B | vLLM NVFP4 | **10/10** | — |
| **Gemma-4-E4B (4.6 GB)** | GGUF | **10/10** | — |
| Qwen3-Coder-30B-A3B | vLLM NVFP4 | 9/10 | topological sort |
| Qwopus3.6-27B-Coder | vLLM NVFP4 | 9/10 | CSV quoting |
| Qwen3.6-35B-A3B | GGUF | 9/10 | CSV quoting |
| GLM-4.7-Flash | vLLM NVFP4 | **5/10** | 01, 06, 08, 09, 10 |

**How to read this.** Eight of twelve models scored full marks, so *the test is
too easy to rank them*. It answers "does this model write correct Python",
nothing finer. The two ends are still informative: a 4.6 GB model passing
everything is a real result, and GLM-4.7-Flash failing half is too — its
failures were verified as genuine logic errors, not truncation.

GLM's failures: did not strip whitespace before parsing; wrong flatten
behaviour; failed to raise on an unterminated quote; reported a cycle in an
acyclic graph; rejected the valid version string `1.2.3-alpha`.

### Single-stream latency, which is what a coding agent feels

| Model | Platform | Prefill @8k | Decode | TTFT @32k |
|---|---|---:|---:|---:|
| **Qwen3-Coder-30B-A3B** | vLLM | **18 602** | 161 | **3.0 s** |
| Gemma-4-26B-A4B | vLLM | 15 611 | 113 | n/a¹ |
| Qwopus3.6-27B-Coder | vLLM | 7 369 | 42 | 5.4 s |
| Qwen3.6-35B-A3B | GGUF | 4 068 | **167** | 6.1 s |
| Coder-Next REAP-40B | GGUF | 3 126 | 107 | 6.6 s |
| Coder-Next REAP-48B | GGUF | 1 253 | 80 | 16.1 s |
| Coder-Next REAP-60B | GGUF | 541 | 56 | 35.4 s |
| Coder-Next 80B | GGUF | 328 | 54 | **61.3 s** |

¹ Gemma's tokenizer produces more tokens for the same text, so the 32k prompt
exceeded the configured window. Not a defect, a setting.

## Section 2 — Text classification

600 multilingual news items. Best F1 per model across concurrency levels.

| Model | Platform | F1 | Accuracy |
|---|---|---:|---:|
| **Gemma-4-31B** | GGUF | **0.976** | 0.975 |
| **Qwopus3.6-27B-Coder** | vLLM | **0.974** | 0.973 |
| **Gemma-4-26B-A4B** | vLLM | **0.969** | 0.968 |
| Gemma-4-26B-A4B | GGUF | 0.968 | 0.967 |
| Qwen3.6-35B-A3B | GGUF | 0.964 | 0.964 |
| Gemma-4-E4B | GGUF | 0.962 | 0.961 |
| Coder-Next 80B | GGUF | 0.957 | 0.955 |
| GLM-4.7-Flash | vLLM | 0.951 | 0.948 |
| Coder-Next REAP-60B | GGUF | 0.931 | 0.932 |
| Coder-Next REAP-48B | GGUF | 0.900 | 0.903 |
| Coder-Next REAP-40B | GGUF | 0.864 | 0.867 |

### By language — where models actually differ

| Model | Total | ru | ua | **lt** | pl | .news | .com |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwopus-27B | 0.974 | 0.972 | 0.992 | 0.944 | 1.000 | 0.982 | 0.969 |
| Qwen3.6-35B | 0.959 | 0.957 | 0.968 | **0.971** | 1.000 | 0.982 | 0.935 |
| Coder-Next 80B | 0.957 | 0.967 | 0.984 | 0.900 | 0.909 | 0.972 | 0.935 |
| Qwen3-Coder-30B | 0.936 | 0.947 | 0.952 | 0.773 | 0.909 | 0.963 | 0.944 |
| REAP-60B | 0.931 | 0.960 | 0.934 | 0.833 | 0.909 | 0.962 | 0.891 |
| REAP-48B | 0.900 | 0.918 | 0.936 | 0.788 | 0.909 | 0.918 | 0.852 |
| REAP-40B | 0.864 | 0.895 | 0.871 | **0.686** | 0.909 | 0.857 | 0.874 |

Russian and Ukrainian are handled well by everything, 0.87 to 0.99. **Lithuanian
is the discriminator**, spanning 0.686 to 0.971. Judge a model on the small
language, not on the average. The Moldovan subset is omitted: it contains no
relevant items, so F1 is undefined there.

### Throughput against concurrency

Articles classified per second. This is the largest effect in the whole study.

| Model | Platform | c=1 | c=8 | c=16 | c=32 | c=64 | Gain |
|---|---|---:|---:|---:|---:|---:|---:|
| **Gemma-4-26B-A4B** | vLLM | 9.1 | 57.3 | — | 156.7 | **159.6** | **×17.6** |
| **Qwen3-Coder-30B** | vLLM | 12.2 | 56.2 | — | 119.0 | **129.0** | **×10.5** |
| **GLM-4.7-Flash** | vLLM | 10.2 | 34.7 | — | 81.6 | **99.7** | **×9.8** |
| Qwopus-27B | vLLM | 3.5 | 17.6 | — | 30.2 | 30.4 | ×8.8 |
| Gemma-4-E4B | GGUF | 10.9 | 25.2 | 35.2 | — | — | ×3.2 |
| Gemma-4-26B-A4B | GGUF | 5.8 | 8.0 | 8.1 | — | — | ×1.4 |
| Qwen3.6-35B | GGUF | 7.1 | 8.7 | 10.0 | — | — | ×1.4 |
| REAP-40B | GGUF | 4.8 | 6.5 | 7.8 | — | — | ×1.6 |
| REAP-48B | GGUF | 3.0 | 4.0 | 4.4 | — | — | ×1.5 |
| REAP-60B | GGUF | 1.8 | 2.1 | 2.3 | — | — | ×1.3 |
| Coder-Next 80B | GGUF | 1.4 | 1.6 | 1.7 | — | — | ×1.2 |
| Gemma-4-31B | GGUF | 1.5 | 1.6 | fails² | — | — | ×1.1 |

² Gemma-4-31B is dense, so its per-slot context cache is large; sixteen parallel
slots do not fit in 32 GB. It runs with eight.

Quality does not change with concurrency — F1 stays inside its noise band at
every level.

## Section 3 — Translation

Six articles into seven languages, 42 translations, scored with chrF++ against
the reference set.

| Model | Platform | chrF++ | RO | DE | FR | ES | PL | UK | RU | Wall |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Qwopus-27B** | vLLM | **70.11** | 67.4 | 70.6 | 77.1 | 77.7 | 65.5 | 65.0 | 67.4 | 157 s |
| **Gemma-4-31B** | GGUF | **70.11** | **69.7** | 71.6 | 77.2 | **78.4** | 65.8 | 64.1 | 63.9 | 527 s |
| **Gemma-4-26B-A4B** | vLLM | **69.68** | 68.7 | 70.6 | **77.6** | 76.8 | 65.5 | 63.8 | 64.9 | **78 s** |
| Gemma-4-26B-A4B | GGUF | 69.51 | 67.9 | 70.8 | 77.1 | 76.6 | 64.7 | 64.6 | 64.8 | 121 s |
| Qwen3.6-35B | GGUF | 69.34 | 69.2 | **71.7** | 74.6 | 76.8 | 64.0 | 63.7 | 65.5 | 113 s |
| GLM-4.7-Flash | vLLM | 67.00 | 66.2 | 67.7 | 75.2 | 74.8 | 61.4 | 59.9 | 63.7 | 97 s |
| Gemma-4-E4B | GGUF | 66.66 | 66.3 | 68.8 | 74.3 | 76.8 | 62.1 | 55.5 | 62.9 | 76 s |
| Qwen3-Coder-30B | vLLM | 61.81 | 62.0 | 62.2 | 68.2 | 71.8 | 55.1 | 54.3 | 59.1 | 73 s |

**How to read this.** The spread from best to worst general model is 3.4 points,
and the bottom of that range is a 4.6 GB model. Romance languages score 74–78
everywhere, Slavic ones 55–67 everywhere: **the language matters more than the
model.** The clear outlier is the coding-specialised model at 61.8 — confirming
that a code model is the wrong tool for prose.

Mechanical checks: 6 to 13 numbers lost out of roughly 500 per model, and 2 to
27 English-word hits — nearly all of them proper nouns correctly preserved. One
genuine defect: **Gemma-4-31B left an entire footer line untranslated** in the
French version of one article. All five other models translated it.

## Section 4 — Load and unload

Cold means the page cache was dropped on the host first. Warm means it was not.

| Model | Platform | Cold | Warm | Unload | VRAM |
|---|---|---:|---:|---:|---:|
| Gemma-4-E4B | llama.cpp | 6.2 s | **2.4 s** | 0.03 s | 3 902 |
| Gemma-4-26B-A4B | llama.cpp | 16.5 s | 4.7 s | 0.04 s | 18 184 |
| Gemma-4-31B | llama.cpp | 18.4 s | 5.2 s | 0.04 s | 24 614 |
| Qwen3.6-35B | llama.cpp | 21.2 s | 5.2 s | 0.04 s | 21 880 |
| REAP-40B | llama.cpp | 25.6 s | 6.1 s | 0.05 s | 28 160 |
| REAP-48B | llama.cpp | 30.2 s | 6.5 s | 0.04 s | 30 834 |
| REAP-60B | llama.cpp | 36.9 s | 6.8 s | 0.05 s | 30 704 |
| Coder-Next 80B | llama.cpp | 44.7 s | 7.1 s | 0.04 s | 30 562 |
| GLM-4.7-Flash | vLLM | 50.6 s | 39.8 s | 0.38 s | 28 350 |
| Qwen3-Coder-30B | vLLM | 62.2 s | 36.4 s | 0.34 s | 29 740 |
| Qwopus-27B | vLLM | 86.2 s | 70.8 s | 1.14 s | 27 382 |
| Gemma-4-26B-A4B | vLLM | 140.1 s | **116.1 s** | 0.82 s | 28 842 |

llama.cpp reaches serving in two to seven seconds warm, whatever the model.
vLLM needs 36 to 116 seconds even warm. Unloading is effectively instant on both.

### Where vLLM's startup time actually goes

Measured phase by phase on Gemma-4-26B-A4B, total 117 s. Full log in
`data/vllm-startup-gemma.log`.

| Phase | Seconds |
|---|---:|
| **Multi-modal warmup** | **51** |
| torch.compile artifact reconstruction | 21 |
| Engine init and configuration | 14 |
| **Loading the weights** | **9** |
| Distributed init | 3 |
| Chat template detection | 3 |
| FlashInfer autotune | 2 |
| **CUDA graph capture** | **2** |

Two surprises. Reading 18 GB off disk takes **9 seconds**, not much worse than
llama.cpp — the slow part is everything else. And CUDA graph capture, the usual
suspect, costs **2 seconds**.

### "Multi-modal warmup" — what that actually is

Jargon worth unpacking, because it is half the startup time.

**Multi-modal** means the model handles more than one kind of input. Gemma-4 is
not only a text model: the same file also contains a part that looks at
**images** and a part that listens to **audio**. Three machines in one.

**Warmup** means vLLM does not simply read the weights and declare itself ready.
It pushes a few fake requests through the model first, to learn how much memory
each path needs at full size and to make the GPU prepare the code it will run.
**A self-calibration pass.**

It calibrates the image and audio paths too — sending fake pictures and fake
sound through at the largest sizes allowed, measuring as it goes. We only ever
send text. So those 51 seconds tune machinery we never touch.

`--language-model-only` tells it to skip the parts we do not use:

| | Startup | Throughput @c32 |
|---|---:|---:|
| default | 117 s | 159.8 art/s |
| `--language-model-only` | **46–57 s** | 158.5 art/s |

**Half the startup time, no measurable cost.**

## Section 5 — CUDA graphs, since the question comes up

All four measured twice; only the second run of each is comparable, because
changing a compilation flag invalidates the compile cache.

| Configuration | Startup | c=1 | c=32 |
|---|---:|---:|---:|
| default | 117 s | **9.15** | **159.8** |
| `--max-cudagraph-capture-size 256` | 130 s | 9.15 | 158.8 |
| `--cudagraph-capture-sizes 1 2 4 8 16 32` | 128 s | 9.16 | 150.7 |
| `--enforce-eager` | 113 s | **2.48** | **69.9** |

**Leave them alone.** Turning graphs off saves 4 seconds of startup and costs
73% of throughput at one request and 56% at thirty-two. The capture-size flags
take token-batch sizes, not request counts, so restricting them pushes real work
onto the slow path for no gain.
