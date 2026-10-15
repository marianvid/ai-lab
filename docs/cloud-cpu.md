# Cloud & CPU — models that never touch the card

Some models are served through the same address as the others but never use
the graphics card:

- **Subscription models** (`claude/sonnet`, `claude/opus`, `claude/haiku`, …):
  answered in the cloud by the vendor, through its command line program
  (Claude Code, Codex, opencode) run on this machine's processor.
- **Processor models** (`bge-m3`): run on the processor, from RAM. Today they
  make embeddings — a vector per text, compared with others to tell when two
  texts say the same thing, in the same language or not. One that is allowed
  to may borrow the card while it has room, and steps back to the processor
  whenever a card model is loaded ([ADR 0004](decisions/0004-embeddings-borrow-the-card.md)).

They take no place on the card, never wait for a model to be loaded and never
hold one up. Each has a queue of its own: one per vendor for subscriptions,
one per model on the processor. That is why they have a page of their own:
nothing on the [Gateway](gateway.md) page — loads, evictions, the card's
queue — applies to them.

## The page

**Server address** — the base URL and the paths these models answer:
`POST /v1/chat/completions` for subscriptions, `POST /v1/embeddings` for
processor models. How to write the requests is in
[Writing a request](requests.md#models-that-never-touch-the-card).

**Subscription · vendor**, one panel per vendor:

| Figure | Meaning |
|---|---|
| Models | The names to put in `model`. |
| Status | Idle, working, or paused: after the vendor says its allowance is spent, every call to that vendor waits, then tries again. |
| Processing | Calls running now, out of how many may run at once for this vendor. |
| Default effort | How long the model thinks when a request names no `reasoning_effort`. |
| Today | Calls since midnight, and how many were answered. |
| Failures today | Counted by the reason the CLI gave. |

**Embeddings · model**, one panel per model:

| Figure | Meaning |
|---|---|
| Status | Running or stopped. The first request that names it starts it; it stops on its own when unused for the idle limit. |
| Answers on | Where the next request goes: the card (and how many requests are there now), or the processor. A model allowed on the card goes there when it has room and leaves whenever a card model is loaded. |
| Processing / Queued | Requests at the model now, out of its places; and those waiting for a place. |
| Requests | Since the manager started, and how many failed. |
| Idle / Stops after | Seconds since the last request; how long it may sit unused before it gives its RAM back. |
| Start / Stop | Do it now rather than wait. Stop is refused while requests are being answered. |

The page refreshes itself every five seconds, like the Gateway page: the
traffic comes from outside the manager and produces no events.

## Configuring them

Subscription models: the `providers` section of `config.json`, described in
[ADR 0001](decisions/0001-subscription-providers.md) and
[ADR 0002](decisions/0002-opencode-provider.md). Processor models: the
`cpu_models` section, described in
[ADR 0003](decisions/0003-processor-models.md), with the measurements behind
its defaults (four places, 2,048 tokens per text, stop after 30 minutes idle).
A `card` entry lets a model borrow the card:
`"card": {"port": 8391, "needs_mb": 1300}` — its own port on the card, and the
card memory it was measured to take; it goes there only with that much plus
`margin_mb` (500 by default) free. The rule is in
[ADR 0004](decisions/0004-embeddings-borrow-the-card.md).
