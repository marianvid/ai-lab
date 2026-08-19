# Five ways the measurements lied

Every one of these produced numbers that looked entirely convincing and were
wrong. They are recorded because the next person — quite possibly the same
person with no memory of this — will hit them again.

The common shape: **the harness was broken, and the model got the blame.**

---

## 1. A reasoning model scored zero because it was cut off mid-thought

**What it looked like.** Qwopus3.6-27B-Coder classified nothing. F1 0.000,
articles answered 0 of 600, at every concurrency level. On the coding test it
scored 7/10 with two failures reading `SyntaxError: unterminated string literal`.
The obvious reading: a broken or useless model.

**What was happening.** Qwopus reasons before answering. The token budget was
`40 × batch + 60`, about 260 tokens, which was generous for the answer and far
too small for the thinking that preceded it. The model never reached the JSON.
The truncated code, likewise, was simply cut in half.

**The fix.** Ask the server to skip the reasoning pass with
`chat_template_kwargs: {"enable_thinking": false}`, retry without it if the
server rejects the argument, strip any `<think>` block that survives, and raise
the budget.

**After the fix, Qwopus had the best classification score of any model tested:
F1 0.974.** The gap between "useless" and "best in class" was one parameter of
mine.

**Lesson.** If a model scores zero, suspect the harness before the model. A real
model is rarely *exactly* zero.

---

## 2. Parallel slots silently shrank the context window

**What it looked like.** Latency results for every GGUF model came back with
`null` for the 8k and 32k prompts. HTTP 400, Bad Request.

**What was happening.** The server was started with `--parallel 16` so that the
concurrency test would have slots. llama.cpp divides `--ctx-size` among the
slots: 32768 / 16 = 2048 tokens each. The long prompts did not fit and were
rejected.

**The fix.** Two starts per model: one slot for latency, many slots for
throughput.

**Lesson.** In llama.cpp, `--parallel` and `--ctx-size` interact. The context you
asked for is not the context a request gets.

---

## 3. The clock watched the wrong field

**What it looked like.** The production model, Qwen3.6-35B, appeared to generate
at 32–70 tok/s and never to produce a first token at all — `ttft` was `null` on
every prompt.

**What was happening.** The model streams its reasoning on a separate
`reasoning_content` field. The timer only watched `delta.content`, so it never
saw a first token, `ttft` stayed `None`, and the decode rate was computed over
the whole request including prefill.

**The fix.** Count either field as output, and disable thinking where possible.

**Corrected, the same model measured 4 068 tok/s prefill and 167 tok/s decode —
the fastest decode of anything tested.** It had looked like one of the slowest.

**Lesson.** When streaming, know every field the server can put tokens in.

---

## 4. A comma read as an error

**What it looked like.** The translation checker reported **92 numbers lost**
across 42 translations. That reads like a model mangling figures, which for news
copy would be disqualifying.

**What was happening.** English writes `4,475`. German and Romanian write
`4.475`. French and Russian write `4 475`. The check compared digit strings
literally, so correct localisation counted as data loss.

**The fix.** Strip separators that sit between digits, then compare bare digit
runs.

**Real figure: 9 lost numbers out of roughly 500.**

**Lesson.** A mechanical check needs to know the conventions of the languages it
is checking, or it measures formatting and calls it accuracy.

---

## 5. The first run after a start is 40% slow

**What it looked like.** The same configuration measured 94.1 articles/s in one
pass and 159.8 in another. A 70% spread with nothing changed. Worse, this nearly
produced a false conclusion: `--language-model-only` appeared to cost 40% of
throughput.

**What was happening.** The first benchmark after a server start runs while the
kernel autotune cache is still settling. `bench.py` did an untimed warm-up;
`bench_classify.py` did not. Whichever test ran first after a start paid the
penalty.

Proof, two consecutive identical runs against one server:

| | articles/s |
|---|---:|
| first | 94.0 |
| second | 158.5 |

**The fix.** `bench_classify.py` now burns one untimed pass before timing.

**Corrected, `--language-model-only` costs nothing: 158.5 against 159.8.**

**Lesson.** Warm up before every measurement, not just the one where you thought
of it.

---

## A sixth, about vLLM specifically

**Changing any compilation-related flag invalidates vLLM's on-disk compile
cache.** The first start with a new flag therefore pays a full recompilation and
is not comparable to a steady-state start.

This produced a backwards result: `--max-cudagraph-capture-size 256` appeared to
make startup *slower* (222 s against 135 s) when it was simply the only
configuration paying for a rebuild. Measured with each configuration holding its
own warm cache, the real spread was 113–130 s.

**Always start twice with a new flag and use the second number.**

---

## The rule that would have caught all of them

Before believing any result, ask: **if my harness were broken in the most likely
way, would the output look exactly like this?**

Zero is suspicious. So is a perfectly round failure, an unexplained 70% spread,
and any number that changes a conclusion. Every one of the six above was found
by looking at the actual output — the raw text, the server log, the tensor
shapes — rather than at the summary score.
