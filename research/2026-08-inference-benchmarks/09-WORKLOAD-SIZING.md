# How big is the actual work?

Measured while sizing an inference engine. The engine question dissolved — see
`08-ENGINE-MODEL-SUPPORT.md` — but the data is about the corpus, not the engine,
and stays useful for any of them.

Scripts: `benchmark/harness/measure_articles.py` and `tokratio.py`.
Raw output: `data/article_lengths.json`.

## Why it was measured

The plan was to move bulk text processing from cloud models onto this card. The
ParallaxVox pipeline currently triages on headlines alone, because when it was
built there was no GPU and prompt reading was expensive. With enough prefill
speed the model can read the whole article instead — which changes every size
in the system.

The existing pipeline numbers are no guide. They are the scar of not having a
card, not a statement of what the work needs.

## Real article lengths

226 articles fetched successfully from a 240-URL sample of the ParallaxVox pool,
stratified across languages. Extracted with trafilatura, **no character cap** —
the pipeline's own extractor truncates at 4 000 characters, and every gold set is
already cut at 1 800, 2 000 or 3 500, so none of the stored data shows natural
length.

| | characters | paragraphs |
|---|---:|---:|
| median | 1 929 | 10 |
| p75 | 3 302 | 16 |
| p90 | 4 719 | 26 |
| p99 | 10 396 | 53 |
| longest of 226 | 13 684 | 114 |

By language, in characters:

| | median | p90 | max |
|---|---:|---:|---:|
| .com | 2 430 | 6 642 | 8 074 |
| .pl | 2 005 | 5 561 | 11 308 |
| .ua | 1 917 | 3 599 | 13 684 |
| .lt | 1 404 | 4 719 | 8 452 |
| .ru | 1 324 | 4 440 | 7 563 |
| .news | 952 | 2 703 | 4 177 |

**94% of the July links still resolved** — 14 failures out of 240: five 404, two
410, one 504, one connection error, five with no extractable text. Worth knowing
before planning any re-collection.

## Characters per token, by language and tokenizer

The missing link between character counts and engine limits. Measured on the
600-item gold set with each model's own tokenizer.

| Language | Gemma-4 | Qwen3-Coder |
|---|---:|---:|
| Russian | **3.73** | 2.76 |
| Ukrainian | **2.89** | 1.99 |
| Lithuanian | **2.83** | 2.32 |
| Polish | **3.29** | 2.81 |
| English | 5.05 | 4.95 |
| **whole corpus** | **3.54** | **2.75** |

Same text: 43 803 tokens for Gemma, 56 250 for Qwen. **Gemma's tokenizer is 28%
more efficient on this corpus, and 45% better on Ukrainian.**

For bulk Slavic-language work that is a real advantage independent of quality:
fewer tokens per article means more articles per second, a smaller KV cache per
article, and therefore more articles in flight at once.

(The Qwen figure came from the Qwen3-Coder-30B tokenizer, not from the
Qwen3.6-35B in production. Same family, very probably identical, not verified.)

## What an article costs, in tokens

Applying each language's ratio to each fetched article, for Gemma:

| | whole article | full request¹ | three quarters¹ |
|---|---:|---:|---:|
| median | 518 | 1 088 | 959 |
| p90 | 1 251 | 1 821 | 1 508 |
| p99 | 3 160 | 3 730 | 2 940 |
| max | 4 735 | 5 305 | 4 121 |

¹ plus a 400-token system prompt and 170 for title and description.

Coverage against a context limit:

| limit | articles that fit whole |
|---:|---:|
| 2 048 | 94.7% |
| 3 072 | 98.2% |
| 4 096 | 99.6% |
| 6 144 | 100% |

## Two conclusions that change the design

**Truncating by paragraph is not worth it.** At the median the whole article
costs 518 tokens and three quarters costs 389. You save 129 tokens and lose a
quarter of the context. Truncation only matters in the tail — so cap by
characters at a level that bites only there. **6 000 characters keeps 95% of
articles whole** and shortens only the long ones.

**Stop packing many articles into one prompt.** The current pipeline sends 50 per
request, with `max_tokens = 50 × 100` and a concurrency of 6. That was the right
shape for a slow serial engine, where the fixed cost of a request had to be
amortised.

With continuous batching the engine does that better, at token granularity.
One article per request, many requests in flight:

- a malformed answer loses one article, not fifty
- **the whole retry ladder in `core/local_classify.py` exists because a batch of
  50 can parse partially.** One per request removes that failure class
- KV cache frees as each article finishes, instead of being held until the
  fiftieth is done
- the scheduler, not the caller, decides what runs together

## The envelope, for whatever engine

Sized from the measurements above, with margin:

| | value | reasoning |
|---|---:|---|
| context per request | **8 192** | covers ~29 000 Cyrillic characters; double the longest seen |
| output per request | ~150 | a JSON verdict, not prose |
| concurrency | as high as VRAM allows | this is the throughput lever |

8 192 rather than the 4 096 that would cover 99.6%, because AI-Lab is meant to
grow into a text *and image* processing centre, and re-deciding a context limit
should not require re-planning.

For reference, production currently runs `llm-qwen` with `ctx_size 98304` split
across `parallel 4`, so 24 576 per slot. Three times more than the measurements
call for.

## What is still unknown

- **Prompt cost for image work.** Everything here is text. AI-Lab is intended to
  process images too, and nothing about that has been measured.
- **How much of an article is actually needed** to judge relevance or
  orientation. The measurements say the whole article is cheap, so the question
  is now about accuracy rather than cost — and it has not been tested.
- **Whether classification improves** when the model sees the article instead of
  the headline. That is the whole premise of moving this work local, and it is
  an experiment nobody has run: same gold set, headline-only versus
  headline-plus-body, same model.
