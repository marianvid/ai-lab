# Four engines, measured

`08-ENGINE-MODEL-SUPPORT.md` argued from documentation that no third engine was
worth adding. That was half a conclusion: TensorRT-LLM and SGLang were dismissed
partly on published numbers from other hardware — the exact mistake catalogued in
`04-PITFALLS.md`.

Both were then actually run. The conclusion holds, but now it is measured, and
two of the published claims turned out to be wrong in opposite directions.

## TensorRT-LLM against vLLM — Qwen3-Coder-30B-A3B

The only model of the six here that released TensorRT-LLM can load.

| Concurrency | TensorRT-LLM | vLLM |
|---:|---:|---:|
| 1 | 10.8 | **12.3** |
| 8 | 44.4 | **56.3** |
| 32 | **120.5** | 119.1 |
| 64 | 115.1 | **129.0** |
| **peak** | 120.5 | **129.0** |

| | TensorRT-LLM | vLLM |
|---|---:|---:|
| Prompt reading, 8k | 17 217 tok/s | **18 602** |
| Generation | 139.5 tok/s | **161** |
| Coding | **10/10** | 9/10 |
| Classification F1 | 0.924 | **0.936** |
| Startup | 67.6 s | 62.2 s cold / 36.4 warm |

TensorRT-LLM saturates at 32 concurrent requests and declines after; vLLM keeps
climbing to 64. It leads at exactly one point on the curve, by 1%.

**The published claim of 10–25% higher throughput at high concurrency does not
hold on workstation Blackwell.** Those figures were measured on H100/H200/B200.
A likely reason surfaced during the search: **`sm_120` does not support
FlashAttention-4**, so the newest attention work targets datacentre parts.

### What it cost to get one number

Ten walls, documented in `08-ENGINE-MODEL-SUPPORT.md`. The last two:

- **Released 1.2.1 refuses `compressed-tensors` NVFP4**:
  `NotImplementedError: Unsupported quantization_config: 'nvfp4-pack-quantized'`.
  It reads NVFP4 only in NVIDIA's own **`modelopt`** packaging. A third copy of
  the same weights had to be downloaded.
- The trap was symmetrical and almost comic: **Qwen3-Coder-30B had the supported
  architecture but the wrong quantisation packaging; Qwen3.6-35B had the right
  packaging but an unsupported architecture.** Each was missing precisely the
  half the other had.

## SGLang against vLLM — Gemma-4-26B-A4B NVFP4

Same model, same weights, same prompts.

| Concurrency | SGLang | vLLM |
|---:|---:|---:|
| 1 | 8.5 | **9.1** |
| 8 | 47.1 | **57.3** |
| 32 | 71.0 | **156.7** |
| 64 | 101.3 | **159.6** |
| **peak** | 101.3 | **159.6** |

| | SGLang | vLLM |
|---|---:|---:|
| Prompt reading, 8k | 10 419 tok/s | **15 611** |
| Generation | 100.0 tok/s | **113** |
| Coding | 10/10 | 10/10 |
| Translation chrF++ | 69.59 | 69.68 |
| Classification F1 | 0.971 | 0.969 |
| Flags needed to start | **two** | none |

**vLLM leads on every speed measure** — 58% on peak throughput, 50% on prompt
reading, 13% on generation.

**Quality is identical on all three tasks.** Both score 10/10 on code. The
translations match to within 0.09 chrF++ overall and across all seven languages
individually:

| | mean | ro | de | fr | es | pl | uk | ru |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SGLang | 69.59 | 68.6 | 70.7 | 76.6 | 76.8 | 65.4 | 64.1 | 65.0 |
| vLLM | 69.68 | 68.7 | 70.6 | 77.5 | 76.8 | 65.5 | 63.8 | 64.9 |

The mechanical checks agree too: 8 English-word hits each, 7 against 6 lost
numbers. **The engine sets the speed; the weights set the quality.** That is
worth knowing in itself — it means an engine comparison never needs to re-argue
quality.

**And the published claim in the other direction was also wrong.** A head-to-head
found during research reported vLLM ahead by 3× on Gemma-4. Measured here it is
1.6×. Overstated by roughly double — which is why it should not have been used
to skip the test in the first place.

### Friction, which no benchmark table shows

SGLang needed two adjustments to start a model vLLM starts with defaults:

- **`--mem-fraction-static 0.80`.** At 0.90 — the value vLLM runs happily at —
  it ran out of memory after 368 seconds of loading. The two engines account for
  reserved memory differently, so settings do not transfer between them.
- **`--moe-runner-backend flashinfer_cutlass`.** The default MoE runner does not
  support NVFP4: *"Unsupported moe_runner_backend for NVFP4 MoE. Use
  flashinfer_cutlass instead."*

Neither is a defect, and both error messages named their own fix — unlike
TensorRT-LLM, where the walls had no way through. But it is real setup cost.

## The production model gains most of all

Separately, Qwen3.6-35B-A3B — the model actually in production — was measured in
NVFP4 on vLLM against its GGUF form on llama.cpp. Same weights, two formats.

| | NVFP4 / vLLM | GGUF / llama.cpp |
|---|---:|---:|
| Prompt reading, 8k | **13 916** tok/s | 4 068 |
| Generation | **201.5** tok/s | 167 |
| First token at 32k | **2.41 s** | 6.06 s |
| Coding | **10/10** | 9/10 |
| Translation chrF++ | 69.33 | 69.34 |
| Classification F1 | **0.968** | 0.959 |
| **Peak throughput** | **47.1** items/s | 10.0 |

**4.7× the throughput, 3.4× faster prompt reading, and the highest generation
speed measured anywhere in this study — 201.5 tok/s.** Translation quality
identical, classification marginally better.

### Why it scales less than Gemma

Qwen3.6 reaches ×4.4 from concurrency where Gemma-4-26B reaches ×17.6. The
reason is memory, and it is measurable:

| | weights | KV cache |
|---|---:|---:|
| Gemma-4-26B | 18.8 GB | **336 454** tokens |
| Qwen3.6-35B | 23 GB | **252 781** tokens |

Four more gigabytes of weights costs 84 000 tokens of cache. **A larger model is
not only slower per token; fewer requests fit at once.** On a 32 GB card that
second effect dominates at high concurrency.

## Where this leaves the engine choice

| Engine | Model coverage | Peak throughput | Starts on defaults |
|---|---|---:|---|
| **vLLM** | 6 of 6 | **159.6** | yes |
| SGLang | all four types known | 101.3 | no — two flags |
| llama.cpp | yes | 8.1 | yes |
| TensorRT-LLM | 1 of 6 | 120.5¹ | no — ten walls |

¹ measured on a different model; not directly comparable.

The recommendation is unchanged, but it is now evidence rather than inference:

- **vLLM** where throughput matters. It leads every engine on every speed
  measure, and it is the only one that ran all six models.
- **llama.cpp** where startup and coverage matter — seconds instead of a minute,
  and the only engine that moves computation to the CPU rather than streaming
  weights across the OCuLink link.
- **SGLang** is a real, working engine and a legitimate peer. It simply gives no
  reason to switch.
- **TensorRT-LLM** is not usable here for anything current.

## The lesson worth keeping

Two published comparisons were checked against measurement. One overstated
vLLM's advantage by 2×; the other promised TensorRT-LLM an advantage that does
not exist on this hardware at all.

**Neither was usable, and they erred in opposite directions.** That is the case
for publishing a harness rather than a leaderboard.
