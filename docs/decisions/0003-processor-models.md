# 0003 — Processor models: embeddings in a lane of their own

Status: accepted, 2026-10-14

## Context

A client needs embeddings (one vector per text, compared by cosine) to find
the same story told twice, across languages. An embedding model is hundreds
of times smaller than a chat model and makes one pass over the text, so it
can run on the processor, from RAM, and leave the card to the models that
need it.

Measured on CT 102 (Ryzen 7 PRO 8845HS, 12 cores given to the container),
llama.cpp on the processor only, texts of about 300 tokens:

| Model | RAM | One text | Throughput (4 at once) | Same story vs other story (cosine) |
|---|---|---|---|---|
| bge-m3 F16 | 2.8 GB | 0.48 s | 3.0 texts/s | same ≥ 0.81 (ro, ja, zh, de, retold), other ≤ 0.56 |
| bge-m3 Q8_0 | 2.3 GB | 0.54 s | 2.4 texts/s | the same as F16 within 0.003 |
| multilingual-e5-small Q8_0 | 1.4 GB | 0.16 s | 29 texts/s | other stories score above the same story told in Japanese: unusable |

## Decision

1. Processor models are served by `llama-server --embedding`, with
   `CUDA_VISIBLE_DEVICES` empty and `-ngl 0`: the card is never touched.
2. They are a lane of their own (`ai_lab/cpu_models/`, `CpuLane`), beside the
   card's scheduler like the subscription providers (ADR 0001). A request
   never takes a card lease. Each model has its own queue: at most
   `parallel` requests reach its server, the rest wait in the lane.
3. The manager starts a model's server itself, as a child process, on the
   first request (about 1.5 s), and stops it after `idle_stop_s` without use
   (default 30 min; 0 keeps it running). `POST /api/cpu-models/{name}/start`
   and `/stop` do the same by hand. The servers stop with the manager.
4. `POST /v1/embeddings` on the gateway, OpenAI shape. Models, ports and
   limits are configuration (`cpu_models` in `config.json`).
5. bge-m3 F16 is the first model: it separates a retold story from a
   different one in every language tried, and F16 is faster here than Q8_0.

## Consequences

- One text may have at most `context / parallel` tokens (2048 with the
  defaults); a longer one is refused with the server's own 400. With
  `"kv_unified": true` the places share one pool of `context` tokens instead:
  a text may use up to the whole pool (the model's own limit is 8,192), and
  more places cost no more memory. Raising the pool
  costs dearly on the processor: 4 × 8192 held 8.5 GB of RAM and embedded one
  text of 2,800 tokens in 13 s.
- Short texts are fastest in one large call: 300 sentences (8,100 tokens) took
  10.8 s as one call of 300, and about 20 s split into 4 calls at once.
- Only embeddings for now; the lane refuses any other path by name.
- A model's state, queue and counts are in `GET /api/gateway` under
  `cpu_models`.
- Throughput is a few texts a second: a run of a few hundred articles takes
  one to two minutes. Enough for the volume measured; a smaller model is
  not a way out (see e5-small above).
