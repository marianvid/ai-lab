# Writing a request

Everything below is about the request itself: the two shapes that are accepted,
the one field that is this project's own, and what comes back when a model
cannot be served.

### Two ways of writing a request

Nearly every client speaks the shape above. A client written against
Anthropic's own library speaks a different one and posts to `/v1/messages`
instead. Both are accepted here.

Not every engine answers both. vLLM does; llama.cpp does not. An entry asked
for a shape its engine cannot answer is refused by name, before anything is
loaded, and told which entries would have worked:

```
gemma-general runs on llamacpp, which does not answer /v1/messages.
Configured models that do: coder-fast, glm-flash, qwen36-nvfp4, text-bulk.
```

Which shapes each entry answers is in `GET /v1/models`, so a client can look
rather than guess. The engine declares it, so adding a shape — or an engine
that speaks one — is a line in that engine's file and nothing else changes.

### The `ai_lab` field: asking for a model started a particular way

Some settings go in a request — temperature, top-p, how many tokens to write.
Others decide how the model *process* starts, and cannot: a model has to be
told at startup how much context it will be asked to hold, and how much of the
card it may claim.

Sending one of those in the body does nothing. It reaches the engine, which
does not recognise it and ignores it without a word, and you are left with the
same truncations and no idea why. So they travel in a field of their own:

```json
{
  "model": "coder-fast",
  "messages": [{"role": "user", "content": "..."}],
  "ai_lab": { "context_size": 65536 }
}
```

**Anything the engine declares as a setting can go in there**, and it is
checked by the engine's own rules — the same rules the Settings page uses — so
a name it does not have, or a number out of range, is refused before anything
loads. The ones worth knowing:

| setting | engine | what it decides |
|---|---|---|
| `context_size` | both | how much the model will be asked to hold. The commonest reason to use this field at all |
| `gpu_memory_fraction` | vLLM | how much of the whole card vLLM claims, 0.10 to 0.98. **This is what decides whether a second model fits beside it** |
| `max_sequences` | vLLM | how many requests it interleaves in one pass |
| `parallel` | llama.cpp | slots. llama.cpp divides the context between them rather than sharing it, so four slots means a quarter each |
| `gpu_layers` | llama.cpp | `-1` all on the card, `-2` fit what you can and leave the rest in system memory, a number to set it yourself |
| `cache_type_k`, `cache_type_v` | llama.cpp | how the context cache is stored. `q4_0` costs a fraction of `f16` |
| `tool_calling` | vLLM | which family's tool-call format to expect. Empty means tools are refused |

Two examples of the second row, which is the one that matters for fitting more
than one model:

```json
{ "model": "reviewer",
  "messages": [{"role": "user", "content": "..."}],
  "ai_lab": { "gpu_memory_fraction": 0.45, "max_sequences": 8 } }
```

Half the card instead of nine tenths, so something else can sit beside it —
at the cost of a much smaller cache, and therefore fewer requests at once.

```json
{ "model": "coder-fast",
  "messages": [{"role": "user", "content": "..."}],
  "ai_lab": { "context_size": 98304, "cache_type_k": "q4_0",
              "cache_type_v": "q4_0" } }
```

A long context that would not otherwise fit, bought by storing the cache in
four bits instead of sixteen.

What happens:

- already running that way — the request goes straight through
- running some other way — reloaded with these, then answered
- not running — started with these
- a setting the engine does not have, or a value out of range — refused at
  once, before anything is loaded, naming what is wrong

**The settings are not saved.** One request must not quietly rewrite what you
chose in the page. The entry keeps its own settings, the running model differs
from them until it is unloaded, and the Models page says so when you hover the
row.

A reload takes as long as a reload — 4 seconds for a small model, 13 for a 35B,
about 40 for one on vLLM — so asking for different settings on every step of a
workflow costs that every time. Asking for the same ones repeatedly costs
nothing: the second identical request is answered without reloading.

Nothing here interrupts an answer in progress. Every request takes its place in
turn and holds it to the last byte, so a request wanting different settings
waits like any other. That is why the Load and Unload buttons need a guard and
this does not — the buttons do not queue.

### When a model cannot be loaded

The refusal is the answer, so it carries what an agent needs to correct itself
rather than a sentence to parse. Through `/v1/` it arrives in the shape an
OpenAI client already understands, with the detail beside it:

```json
{
  "error": {
    "message": "there is not enough memory for this model on this machine, whatever else is unloaded",
    "type": "insufficient_memory",
    "code": "model_does_not_fit"
  },
  "ai_lab": {
    "model": "gemma31-nvfp4",
    "engine": "vLLM",
    "needed_mb": 29361,
    "available_mb": 3677,
    "capacity_mb": 32623,
    "pool": "card",
    "loaded": ["text-bulk"],
    "asked": { "context_size": 131072 }
  }
}
```

`needed_mb` against `capacity_mb` says whether it could ever fit here;
`available_mb` says whether it fits now. An agent that reads them can ask again
with a smaller context, or a smaller share of the card, without anybody
guessing.

An engine that refuses for its own reasons passes its own words through
unchanged — vLLM, told to hold more context than it can, names the largest that
would have fitted, and that sentence is the most useful thing in the answer.

**Nothing about any of this is remembered between requests.** This manager
reports what it measures now and works out what a request asks for; knowing
that a particular model and context fitted last Tuesday is the business of
whatever is making the requests.

### Running Claude Code against a model on your own card

This works, and it needs one setting most models do not have on by default.

An agent that uses tools needs the engine to understand how *this* model writes
a tool call. A model does not return structured data when it wants to use a
tool; it writes text, and every model family writes it differently. vLLM has to
be told which to expect, and refuses tool use outright until it is. That is the
**Tool calling** setting on the entry: empty means off, otherwise it names the
model's family — `qwen3_coder`, `gemma4`, `glm47`.

With that set, and a context window big enough for the agent's own prompt:

```sh
ANTHROPIC_BASE_URL=http://ai-lab.lan:8090 ANTHROPIC_AUTH_TOKEN=local ANTHROPIC_MODEL=coder-fast CLAUDE_CODE_MAX_CONTEXT_TOKENS=98304 claude -p "read note.txt and tell me which colour it mentions"
```

Measured here on an RTX PRO 4500 with Qwen3-Coder-30B at NVFP4: Claude Code
sends about 108,000 characters of prompt and tool definitions before your own
question, so 32k of context is not enough. 128k did not fit either — vLLM
wanted 12 GiB of cache and had 10.78 — and said so, naming 117,776 as the
largest that would. 96k fits, loads in 70 seconds, and leaves the card at
29.6 GB of 32.6.

---

[← all documents](../README.md)  ·  [Settings](settings.md)  ·  [Updating an engine](engines.md)
