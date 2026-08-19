# Local inference benchmarks — RTX PRO 4500 Blackwell, August 2026

This directory holds everything from a benchmarking session run on the night of
17–18 August 2026: what was measured, how, the raw results, the mistakes made
along the way, and one real bug found and fixed in vLLM.

It is written so that **someone arriving with no memory of the session can pick
it up and continue**. If that is you, read `opts/00-CONTEXT.md` first — it
names the machines, so it lives in the private half.

## What this was for

Marian owns one NVIDIA RTX PRO 4500 Blackwell with 32 GB of VRAM. The question
was which models to run on it, for three kinds of work:

1. **Coding** — an agent that reads files and writes patches.
2. **Text classification** — deciding which of thousands of news articles matter,
   in Russian, Ukrainian, Lithuanian, Polish and more.
3. **Translation** — English into seven languages, for finished articles.

Everything here is measured on that card. Nothing is copied from a vendor slide.

## How to read this

| File | What is in it |
|---|---|
| `00-CONTEXT.md` | The machines, the software, how to reach them. **Start here** — kept in the private half, `opts/`. |
| `01-METHODOLOGY.md` | What each test measures, how it works, and what the score names mean |
| `02-RESULTS.md` | Every number, in tables |
| `03-FINDINGS.md` | What the numbers mean, and the recommendations that follow |
| `04-PITFALLS.md` | Five ways the measurements lied before they were fixed |
| `05-VLLM-GEMMA4-BUG.md` | A real bug in vLLM 0.27.1, its diagnosis and the patch |
| `06-OPEN-QUESTIONS.md` | What is unfinished, and what to do next |
| `07-PUBLICATION-PLAN.md` | The plan for a public repository and an article |
| `08-ENGINE-MODEL-SUPPORT.md` | Why TensorRT-LLM, MLC and TurboMind were all rejected, and the search for a C++ engine |
| `09-WORKLOAD-SIZING.md` | How long a real article is, in characters and tokens, and what that means for batching |
| `10-FOUR-ENGINES-MEASURED.md` | TensorRT-LLM and SGLang actually run, against vLLM, on this hardware |

| Directory | Contents |
|---|---|
| `benchmark/` | **Everything needed to reproduce this.** Scripts, the exact prompts, the evaluation data, the reference answers. Self-contained — see its own README |
| `data/` | Consolidated results, load times, run logs |
| `raw-results/` | Every individual result file, untouched |

The scripts in `benchmark/` are the real product of this work. The numbers are
just what they printed on one particular night.

## The short version

- **Coding**: `Qwen3-Coder-30B-A3B` in NVFP4 on vLLM. Three seconds to first token
  on a 32 000-token prompt, against sixty-one for the 80B model.
- **Text and translation**: the differences between models are small. The model
  already in production, `Qwen3.6-35B-A3B`, is within noise of the best.
- **The largest single effect measured** was not the model. It was continuous
  batching: vLLM multiplies throughput by up to 17 as concurrency rises,
  llama.cpp by 1.4.
- **Pruned models degrade cleanly and predictably.** The same model cut from 80B
  down scores 0.957 → 0.931 → 0.900 → 0.864.
- A 4.6 GB model solved all ten coding tasks and came within 3.4 points of the
  best translator. Size buys less than expected.
- **No C++ engine supports NVFP4.** The card's native format is served only by
  Python engines. TensorRT-LLM deleted its compiled path, MLC-LLM has not
  supported Gemma-4 in four months, and LMDeploy's C++ engine tops out at INT4.

## One warning

Several numbers in the first drafts of this work were **wrong in ways that looked
convincing**. A model scored zero because the answer was cut off before it
finished thinking. Another looked three times slower because the timer watched
the wrong field. `04-PITFALLS.md` exists so those mistakes are not repeated.
Read it before trusting any new measurement.
