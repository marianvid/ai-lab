# Findings

What the numbers in `02-RESULTS.md` mean, and what follows from them.

## Recommendations

**For a coding agent: `Qwen3-Coder-30B-A3B` in NVFP4, on vLLM.** Three seconds to
first token on a 32 000-token prompt, against sixty-one for the full 80B model.
9/10 on the coding tasks. 17 GB, leaving room for long context. In an agent loop
that re-reads code at every step, prompt reading is the whole experience, and
nothing else comes close.

**For text and translation: probably change nothing.** `Qwen3.6-35B-A3B`, already
in production, scores 0.964 on classification and 69.3 on translation, with the
fastest single-stream generation measured. The best alternatives beat it by 0.012
F1 and 0.8 chrF++ — both inside noise.

**If you do change: `Gemma-4-26B-A4B`.** Best all-rounder in the set. On vLLM it
reaches 159.6 articles/s at concurrency 32; on llama.cpp it starts in 4.7 seconds
warm. Same model, and you can pick the trade-off per situation.

**For bulk pipelines, the engine matters more than the model.** Moving
ParallaxVox-style classification from llama.cpp to vLLM turns roughly forty
minutes of work into about thirty seconds.

## The five things worth knowing

### 1. Continuous batching is the largest effect measured

Not quantisation, not model choice, not the GPU's native formats. vLLM multiplies
throughput by 9.8 to 17.6 as concurrency rises. llama.cpp manages 1.2 to 1.6.

The reason is structural. llama.cpp has a fixed number of slots and divides the
context window among them. vLLM keeps a shared paged pool and interleaves
requests continuously, so a request that finishes early frees capacity
immediately.

Quality is unaffected: F1 stays within its noise band at every concurrency level.

**When this does not apply:** if you send one request at a time, the advantage
mostly disappears — at concurrency 1 the two engines are comparable, and
llama.cpp often wins on latency because it starts far faster.

### 2. Pruning costs judgment, in a straight line

The same model, cut down:

| | F1 | Coding |
|---|---:|---:|
| Coder-Next 80B | 0.957 | 10/10 |
| REAP-60B | 0.931 | 10/10 |
| REAP-48B | 0.900 | 10/10 |
| REAP-40B | 0.864 | 10/10 |

Monotonic, no exceptions. REAP removes the least-used experts permanently, and
the loss shows up on judgment tasks while coding stays at full marks — more
evidence the coding test is too easy, not that pruning is free.

Pruning buys speed: REAP-40B reads prompts ten times faster than the 80B (3 126
against 328 tok/s), because it fits on the card instead of computing on a 35 W
CPU.

**But it is a middle road the data does not recommend.** A native-format model
gives you that speed without cutting anything, and scores better.

### 3. Size buys much less than expected

`Gemma-4-E4B` is 4.6 GB. It solved **all ten** coding tasks, scored 0.962 on
classification, and came within 3.4 chrF++ points of the best translator in the
set. It loads in 2.4 seconds and uses 3 902 MiB.

Across all general models, translation spans 66.7 to 70.1 — 3.4 points, from a
4.6 GB model to a 31 GB one.

**For translation, the language matters more than the model.** Romance languages
score 74–78 for everyone; Slavic ones 55–67 for everyone.

### 4. Dense against mixture-of-experts explains more than parameter count

A mixture-of-experts model has many parameters but activates only a few per
token, so it reads little from memory and generates fast. A dense model reads
everything, every token.

This produces results that look wrong until you know it:

- Coder-Next 80B, half its weights in system RAM, generates at 54 tok/s.
- Qwopus-27B, entirely on the card, generates at 42 tok/s — **slower, while being
  a third the size and fully resident.**

It also decides what fits. Gemma-4-31B is dense, so its per-slot context cache is
large, and sixteen parallel slots will not fit in 32 GB.

### 5. Prompt reading and generation are separate problems

Generation follows memory bandwidth. Prompt reading follows raw compute.

A model that does not fit on the card pays at prompt reading, not at generation —
which is why the 80B model still writes at a usable 54 tok/s yet needs a full
minute before it starts. For interactive work that is the difference between
usable and not.

## Things that turned out not to be true

**"Native formats are smaller."** They are not. NVFP4 and a 4-bit GGUF both spend
about four bits per parameter. Native formats buy *speed*, and mostly the speed
of reading a prompt.

**"The 32 GB card can run any model if you offload."** The binding limit is the
smallest available checkpoint. Qwen3-Coder-Next's smallest native checkpoint is
47.6 GB; no amount of offloading makes NVFP4 fit, and it can only run as GGUF
with experts in system RAM.

**"vLLM is slow to load weights."** It is not. Reading 18 GB takes 9 seconds. The
other 108 seconds are initialisation — half of it warming up vision and audio
towers that a text workload never uses. `--language-model-only` halves the whole
startup at no cost.

**"CUDA graph capture is the startup cost."** It is 2 seconds of 117. Disabling
graphs saves 4 seconds and costs 73% of throughput at one request.

**"An offloaded NVFP4 model is worth trying."** vLLM can do it —
`--offload-backend uva --cpu-offload-gb 24` loads Qwen3-Coder-Next NVFP4 with
20 GB on the card. But it streams weights over the OCuLink link at every step,
and that link is roughly 8 GB/s against the card's 896. llama.cpp avoids this by
moving the *computation* to the CPU instead of the data to the GPU. Not measured;
abandoned as clearly unpromising.

## Worth following up

The ParallaxVox benchmark report records that pass-1 classification moved from
local models to a cloud model for quality. **Qwopus-27B scored F1 0.974 locally
on that project's own gold set** — above the figure recorded there for the cloud
option.

This is a lead, not a conclusion: this test showed the model headline *and*
description, while production pass-1 sees only the headline. The protocols
differ. But it is worth measuring properly, because it would bring that stage
back in-house.
