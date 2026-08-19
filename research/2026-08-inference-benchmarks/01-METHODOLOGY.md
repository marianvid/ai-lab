# Methodology — what was measured, how, and what the words mean

Four measurements. Each is a script in `benchmark/harness/`, and each script is the real
definition of the test; this file explains the intent behind it.

## Vocabulary, before anything else

**Prefill** (also "prompt reading"). Before a model can answer, it must read
everything you gave it. This is one big batch of arithmetic, so it is limited by
raw compute. Measured in tokens per second, it is often in the thousands. It is
what makes a coding agent feel slow when it pastes whole files.

**Decode** (also "generation"). Writing the answer, one token at a time. Each
token requires reading the model's weights out of memory, so this is limited by
memory bandwidth, not compute. Usually tens to low hundreds of tokens per second.

Those two are different bottlenecks, and a model can be fast at one and slow at
the other. Reporting a single "tokens per second" number hides this.

**Time to first token (TTFT).** How long you stare at nothing. For a long prompt
this is essentially the prefill time.

**Concurrency.** How many requests are in flight at once. A server that is fast
for one user can be far faster in total when serving thirty-two, because the GPU
does more useful work per pass. This is the single largest effect we measured.

**F1.** Used for the classification test. When you are sorting items into
"relevant" and "not", two errors are possible: flagging something irrelevant
(hurts *precision*) and missing something relevant (hurts *recall*). F1 combines
them into one number between 0 and 1 by taking their harmonic mean, which
punishes a model that games one at the expense of the other. Accuracy alone is
misleading when the classes are unbalanced; F1 is not.

- **precision** = of everything the model called relevant, how much really was
- **recall** = of everything that really was relevant, how much the model found
- **F1** = 2 × precision × recall / (precision + recall)

**chrF++.** Used for translation. It compares the model's translation to a
reference by counting how many short character sequences they share, plus short
word sequences. Character-level comparison is the right choice for Romanian,
Polish or Ukrainian, where a word can inflect in a dozen ways and a word-level
score would call every inflection a miss. It is computed by `sacrebleu`, needs no
judge model, and cannot have opinions. Range 0–100; higher is closer to the
reference. **It measures similarity to one reference, not quality** — a good
translation phrased differently scores lower than it deserves.

**Cold and warm loading.** Cold means the model file is read from disk. Warm
means the operating system still has it in page cache from a previous load. The
difference is large and worth knowing separately.

## Test 1 — Latency (`bench.py`)

Three prompts of growing size — about 300, 8 000 and 32 000 tokens — each
containing real Python source and asking for a rewrite. Streaming is on, so the
first content token can be timed exactly.

Reports prefill tok/s, decode tok/s and TTFT for each size.

One untimed warm-up runs first. Requests ask the server to skip any reasoning
pass (`chat_template_kwargs: {"enable_thinking": false}`) and fall back
gracefully if the server rejects that. Tokens streamed on a separate
`reasoning_content` field are counted too — see `04-PITFALLS.md` for why both of
those matter.

## Test 2 — Coding (`bench_coding.py`)

Ten Python tasks. **Nothing is graded by opinion.** Each answer is extracted from
its code block, run against a hidden block of assertions, and either passes or
does not.

The tasks cover parsing with error handling, interval merging, fixing a supplied
buggy function, grouping, a decorator factory, recursive flattening, an LRU
cache, CSV quoting rules, topological sort with cycle detection, and semantic
version comparison. Each specifies its error cases, because handling the unhappy
path is where models differ.

Generated code runs inside the container as user `nobody`, in a throwaway
directory, with a 25-second timeout.

**Reference solutions are in `benchmark/reference-solutions/`** and all ten pass.
They exist so the test suite itself is validated — an eval whose tests are wrong
is worse than no eval.

**Known limitation: this test does not discriminate well.** Five models scored
10/10. Treat it as a floor — "does this model write correct Python at all" —
not as a ranking. Harder tasks are needed; see `06-OPEN-QUESTIONS.md`.

## Test 3 — Multilingual classification (`bench_classify.py`)

600 real news items, each a headline plus a short description, from a live
ParallaxVox pipeline run. The model decides whether each is relevant to the
Ukraine–Russia war. Ground truth comes with the set; 52% are relevant.

Language spread, by the site's top-level domain:

| | .ru | .ua | .lt | .news | .com | .pl | .md |
|---|---:|---:|---:|---:|---:|---:|---:|
| items | 154 | 117 | 114 | 99 | 95 | 11 | 10 |
| relevant | 105 | 65 | 18 | 53 | 65 | 6 | **0** |

Two cautions on that table. **Lithuanian is unbalanced** — only 18 relevant out
of 114 — which makes its F1 sensitive and is exactly why weak models collapse
there. And **the Moldovan subset contains no relevant items at all**, so F1
cannot be computed for it; reporting 0.000 would be wrong. Those articles are in
Russian anyway.

Items are sent five per request, so 120 requests, run at a chosen concurrency.
The same run therefore yields both **quality** (F1 per language) and
**throughput** (articles per second, prefill and decode tok/s). That is
deliberate: it is the same work a bulk pipeline does.

The prompt is adapted from the production `phase4_pass1_system.md`, with one
change: production shows the model only the headline, this test also shows the
description, because the ground truth was labelled on headline plus description.
**Scores here are therefore not directly comparable to production numbers.**

## Test 4 — Translation (`bench_translate.py`)

Six English articles translated into seven languages — Romanian, German, French,
Spanish, Polish, Ukrainian, Russian — giving 42 translations per model. Scored
with chrF++ against the reference set.

Two mechanical checks run alongside, because a score can hide a disaster:

- **English left untranslated.** Counts common English function words in the
  output. *This is a weak signal* — most hits turn out to be organisation names
  correctly kept in English, like "Institute for the Study of War". Use it to
  find candidates to read, not as a score.
- **Numbers lost.** Compares the digits in source and translation. Separators
  between digits are stripped first, because `4,475` legitimately becomes
  `4.475` in German and `4 475` in French. Without that normalisation this check
  reported 92 errors where there were 9.

## Rules that apply to every test

- Temperature 0.
- One untimed warm-up before timing. **The first pass after a server start runs
  about 40% slow** while kernel autotune settles; measuring it produces numbers
  that look like a real regression.
- Reasoning turned off where the server supports it, and `<think>` blocks
  stripped from output where it does not.
- Each configuration measured once. **There are no error bars.** Differences
  below roughly 0.02 F1 or 1 chrF++ point are noise.
- Changing any compilation-related vLLM flag invalidates its on-disk compile
  cache, so the first start with a new flag is much slower and not comparable.
  Always start twice and use the second number.
