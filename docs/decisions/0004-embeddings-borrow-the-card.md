# 0004 — Embeddings borrow the card while it has room

Status: accepted, 2026-10-15. Builds on [0003](0003-processor-models.md).

## Context

ADR 0003 put embeddings on the processor so they never cost the card a place.
The processor is slow at it, though, and the card usually has a few GB left
beside its model. Measured on CT 102 (RTX PRO 4500, 32 GB), bge-m3 F16, 300
sentences in one call:

| | Processor | Card |
|---|---|---|
| 300 sentences | 10.8 s | 0.48 s |
| Memory | 2.8 GB of RAM | 0.9–1.3 GB of card memory |
| Start | about 1 s | 1.45 s with the file cached, 1.6–1.7 s from disk |
| Stop | — | 0.26 s |

The vectors agree: cosine between the card's and the processor's vector for
the same text 0.999998; the similarity between two texts differs by at most
0.00047.

The card's own models must not suffer for it. vLLM in particular asks for a
fixed share of the *whole* card at start (`gpu_memory_utilization` × total)
and refuses to start when less than that is free, so anything left on the card
during its start can stop it from loading at all.

## Decision

The card's own models come first; embeddings take the card only when nobody
else needs it.

1. A model with a `card` entry (`{"port": 8391, "needs_mb": 1300}`) may run on
   the card, as a second `llama-server` with every layer there, on its own
   port. Without the entry it stays on the processor.
2. **Room** means `needs_mb` plus a margin (`margin_mb`, 500 MB by default)
   free on the card. Read from the driver; refused on unified memory (a Mac),
   where the card and RAM are the same memory.
3. **A request** goes to the card when it is already there, or when there is
   room to start it there. Otherwise it goes to the processor, and the card is
   asked again at most every 30 seconds after a "no".
4. **Before any card model loads** — through the gateway, the Load button, or
   the restore after a manager restart — the embeddings model leaves the card:
   new requests go to the processor, requests already on the card finish there
   (at most 60 s), then its card server stops. Only then is the card read and
   the model loaded.
5. **After the load**, whether it worked or not, the model comes back to the
   card if there is room and it was used within its idle limit. Otherwise it
   comes back with the next request that finds room.
6. The scheduler counts the card memory the embeddings model holds as free,
   since it is given back on demand: it never evicts a card model to make room
   that the embeddings model would have given up anyway.
7. Each server, on the card or the processor, stops after `idle_stop_s` unused
   (30 minutes). Both may run for a while after a move; the idle one stops.
8. Every answer says where it was made: `"ai_lab": {"device": "card"}` or
   `"cpu"`.

## Consequences

- Embeddings are about 20 times faster whenever the card has room, with the
  same vectors.
- A card load waits for embeddings already on the card to finish (well under a
  second for a typical call) and for the server to stop (0.26 s).
- If the gateway refuses a load after the embeddings model stepped aside (the
  card still too full), nothing reports the load as over; the model stays on
  the processor until a 15-minute safety limit lets it back.
- Requests running on the processor during a move are slower but are never
  refused because of it.
