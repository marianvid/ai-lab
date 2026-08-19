# Open questions and planned work

What is unfinished, in rough order of value.

## Scheduled: the Cohere translation specialist, on the Mac

`CohereLabs/command-a-translate-08-2025` is the only model found that is trained
specifically for translation rather than being a generalist that copes.

It cannot run on the card. It is **dense, 111B**: 222 GB in BF16, 111 GB in FP8,
and **67.1 GB as GGUF Q4_K_M**, against 32.6 GB of VRAM. Being dense, offloading
hurts it far more than it hurts the mixture-of-experts models measured here.

**It fits on the Mac.** M3 Max, 128 GB, llama.cpp build 10449 with Metal — the
same build as the container, so results are comparable. 67 GB is close to the
69 GB GLM-4.5-Air that already runs there.

FP8 and NVFP4 are not options on Apple silicon: Metal has no execution path for
them, whatever the size. No MLX build of this model exists either.

**Expected speed 2–5 tok/s** — calculated from memory bandwidth, not measured.
GLM-4.5-Air reaches 17–18 tok/s there because it is a mixture-of-experts using
about 12B of 106B per token; this one uses all 111B. That means 5–10 minutes per
article and roughly a night for all 42 translations. Acceptable for finished
copy in small volume; useless interactively. Deferred for that reason — it ties
up the machine.

**When it runs, measure:** the same 42 translations with chrF++, the two
mechanical checks, real speed against the estimate, and a **blind side-by-side
read** — the specialist against the best generalist, unlabelled, for Marian to
judge. For finished copy his judgment outranks any automatic score.

`command-r7b` (16.1 GB) fits anywhere but dates from 2024; include only as a
historical baseline.

## The coding test does not discriminate

Eight of twelve models scored 10/10. The test answers "does this model write
correct Python" and nothing finer.

To make it useful: multi-file changes, tasks needing a plan before code,
debugging from a stack trace, tool use, ambiguous specifications requiring a
judgement call. The current suite should stay as a floor.

## No repeated runs, so no error bars

Every configuration was measured once. Differences below roughly 0.02 F1 or
1 chrF++ point should be treated as noise, and are reported that way — but the
noise band was estimated from incidental repeats, not measured. Three runs of
each would fix it.

Two known variance sources are already handled: the first run after a server
start is ~40% slow, and changing a compilation flag forces a rebuild. See
`04-PITFALLS.md`.

## Report the vLLM bug upstream

Diagnosed and patched locally, never reported. Details, reproduction and patch in
`05-VLLM-GEMMA4-BUG.md`. Worth doing because 0.27.1 is the current release, so
there is no version to upgrade to.

## Answer quality was never judged

Everything measured is speed, executed correctness, or similarity to a reference.
Nothing measures whether an answer is *good*: whether the code is idiomatic,
whether the translation reads naturally, whether the classification reasoning
holds up. For translation the blind read above is the plan; for coding there is
no plan yet.

## The ParallaxVox pass-1 lead

Qwopus-27B scored F1 0.974 locally on that project's own gold set, above the
figure its benchmark report records for the cloud model it switched to.

**The protocols differ** — this test showed headline plus description, production
pass-1 sees only the headline. Re-run with headline only, against the same gold,
before drawing any conclusion. If it holds, that stage could come back in-house.

## The hardware comparison for the article

Marian wants a comparison across NVIDIA, AMD, Intel and Tenstorrent. Collected so
far, all from vendor and third-party sources, **none measured here**:

| Card | VRAM | Price | Native formats | vLLM |
|---|---|---:|---|---|
| RTX PRO 4500 Blackwell | 32 GB GDDR7, 896 GB/s | ~$2 000 (~€3 670 server) | FP4, FP6, FP8, INT8/4 | full |
| AMD Radeon AI PRO R9700 | 32 GB GDDR6 | $1 299 | FP8, FP16, INT8 — **no FP4** | ROCm 7.2, official since Jan 2026 |
| Intel Arc Pro B70 | 32 GB | $949 (street ~$1 099) | — | Intel LLM Scaler; no FlashAttention-2, no TensorRT-LLM |
| Tenstorrent Blackhole p150a | 32 GB, pools to 128 GB | $1 299 | — | none; TT-Metal only |

**The published numbers cannot be compared to each other.** Intel reports ~13
tok/s single and 369 tok/s at 50 concurrent on Qwen 27B FP8. AMD reports a 32B
model at 24.9 tok/s and 2 713 tok/s prefill on Vulkan at a 512-token prompt. This
study measured 18 602 tok/s prefill on NVFP4 at 8 000 tokens. Different models,
quantisations, prompt lengths and backends. Putting them in one table would
produce a scientific-looking lie.

**That is the argument for publishing the harness.** A fixed, documented set of
tests anyone can run on their own card turns the comparison into something
legitimate, and lets the repository grow by pull request.

On Tenstorrent the software signal is clear without measuring: getting a model
into production is reported as months on TT-Metal against days on vLLM, and
outside Llama, Falcon and Mistral there is kernel work to do.

**Still needed:** specific Alex Ziskind video links — his channel listing is
behind a YouTube consent wall and cannot be read automatically, but individual
pages can be fetched if given URLs.

## Resolved since this was written

- **A third engine was investigated and rejected.** TensorRT-LLM, MLC-LLM,
  LMDeploy TurboMind and SGLang were all checked. None is worth adding. Full
  reasoning in `08-ENGINE-MODEL-SUPPORT.md`.
- **The vLLM patch is not permanent.** Nightly loads Gemma-4 unpatched and
  measures identically.
- **The workload was measured** rather than assumed: real article lengths,
  characters per token by language, and what that implies for batching. In
  `09-WORKLOAD-SIZING.md`.
- **`--language-model-only` halves vLLM startup** for Gemma-4, from 117 s to
  about 50, at no measurable cost.

## Newly open

- **Does reading the article beat reading the headline?** The premise of moving
  this work local, and untested. Same gold set, headline-only against
  headline-plus-body, same model.
- **Nothing about images has been measured**, and AI-Lab is meant to process them.
- **`--async-scheduling`**, recommended by vLLM's own Gemma-4 recipe, has never
  been tried here.
- **`--limit-mm-per-prompt image=0,audio=0`** may be a cleaner way to skip the
  multimodal warmup than `--language-model-only`. Untested.

## Smaller loose ends

- **Gemma-4-31B could not be measured above 8 concurrent slots.** Dense, so the
  per-slot cache is large. Its concurrency row is incomplete.
- **Gemma's tokenizer** produces more tokens for the same text, so the 32k latency
  prompt overflowed the window. Its TTFT at 32k is missing.
- **Q3 quantisation was dropped** on Marian's instruction, and the data supports
  it: REAP-40B at Q3 measured the same speed as Q4 (3 046 against 3 126 tok/s
  prefill, 108 both) while losing precision. Q3 buys nothing here.
- **The `.md` subset has no positive examples**, so its F1 is undefined. Reported
  as such; do not let a future script emit 0.000 for it.
