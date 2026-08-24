# AI-Lab

> **Built for personal use.** This is a home-lab tool that runs on one
> particular pair of machines. It is public because the measurements and the
> approach may be useful to someone; it is not a product and has no support.
> It may change over time.

_**This was written with an AI agent, and it is meant to be read the same way.**
Take it as a starting point rather than as something to install. Your machine is
not this machine: different card, different amount of memory, different models,
a different idea of what the thing should do. Point your own agent at this
repository and have it adapt the code to what you have. That is a good deal
faster than reading it all yourself, and it is how the code got here._

**A layer that lets an agent workflow use several models on a machine that
cannot hold them all.**

That is the whole point. An agent workflow uses one model to read, another to
write, another to check, etc. A 32 GB card holds one of the large ones, or one large
and one small. Pointed straight at the engines, an agent naming a model that
happens not to be running gets a refused connection and the workflow stops
there.

AI-Lab is one address in front of all of them. A request names any configured
model; if it is loaded the request goes straight through, and if it is not,
**room is made and it is loaded first** — see [the Gateway](#the-gateway-one-address-and-models-that-come-and-go)
for exactly which model comes off and when. As many models stay loaded as the
memory allows, so the common case costs nothing: measured here, two models
served together answered in 0.04 and 0.08 seconds where alternating between
them used to cost a 3-second and a 7-second load.

**One thing is not standard and cannot be.** Some settings decide how a model
*process* starts — how much context it will hold, how much of the card it may
claim — and no chat API has a field for them. They travel in an `ai_lab` object
in the request body, which is this project's specific; a client that does
not send it gets the entry's configured settings and nothing breaks. See
[the `ai_lab` field](#the-ai_lab-field-asking-for-a-model-started-a-particular-way).



Runs on Linux with an NVIDIA card, where systemd supervises the engines, and on
macOS with Apple silicon, where it supervises them itself. llama.cpp works on
both. vLLM works on the Linux machine and is shown greyed out on the Mac, with
the reason — it needs CUDA, and Apple silicon offers Metal.

## Contents

**The four pages**

- [Models](#models-what-is-configured-and-what-is-running) — one row per
  configured model: what it runs, what it can do, whether it is loaded, and how
  long the last load took. Where models are started, stopped and configured.
- [Library](#library-what-is-on-disk-and-what-could-be) — what is on disk, per
  weight format, and a search of Hugging Face to download more.
- [Gateway](#the-gateway-one-address-and-models-that-come-and-go) — the address
  an agent talks to, what is loaded right now, what is queued, and the rules by
  which models are loaded and unloaded.
- [Settings](#settings-the-machine-its-engines-and-where-models-live) — what
  this machine is, how much of its memory models may use, the engines and their
  versions, and where the model store lives.

**Using it**

- [How loading and unloading is decided](#how-loading-and-unloading-is-decided)
  — the queue, who gets unloaded, and why nothing jumps ahead.
- [What fits beside what, measured](#what-fits-beside-what-measured) — every
  model's real footprint on the card. The table to consult before assuming two
  will fit.
- [The `ai_lab` field](#the-ai_lab-field-asking-for-a-model-started-a-particular-way)
  — asking for a model started a particular way, and the settings worth
  knowing.
- [Two ways of writing a request](#two-ways-of-writing-a-request) — the OpenAI
  shape and the Anthropic one, and which engine answers which.
- [When a model cannot be loaded](#when-a-model-cannot-be-loaded) — what the
  refusal contains, so an agent can correct itself.
- [Running Claude Code against it](#running-claude-code-against-a-model-on-your-own-card)
  — the one setting most models need first.
- [Updating an engine](#updating-an-engine) — reading what an update brings
  before taking it, and installing beside what works.

**Working on it**

- [Structure](#structure) — what each module is for. Read `ARCHITECTURE.md`
  before changing anything: it records the one dependency rule and why.
- [Local development](#local-development) and [Deploying](#deploying).
- [Safety model](#safety-model) — what this can and cannot reach.

## Models: what is configured, and what is running

One line per configured model: the name you gave it, the model it runs, and —
at the right, against the buttons — what the model can do, the weight format
and the engine that will serve it. How long the last load took sits on the row
too, because a load runs from four seconds to a minute and by the time you look
back a message elsewhere would be gone. Port, context, temperature, state and
the breakdown of that load by phase live in the tooltip, because they are
wanted occasionally and were costing three lines of screen every time.

![The model list](docs/screenshots/models.png)

**The two small pictures say what the model can do**: a wrench for calling
tools, a photograph for reading pictures. Neither is configured anywhere —
both are read from the model's own files, once, and remembered. A directory of
weights must carry both a vision section *and* a token to put a picture in
before it claims pictures, because a text model's config can name a vision
tower it never uses. For GGUF it is the chat template inside the weights file,
plus the `mmproj-` file beside it that llama.cpp is handed to see with — so the
same model downloaded without that file honestly shows no picture icon.

The weights decide what a model *can* do; a setting can only take something
away. vLLM's "Text only" loads a model that can see without the part that sees,
so the picture icon goes when it is set — and the wrench stays, because that
setting has nothing to do with the chat template.

**Load** and **Unload** act directly, not through the queue, so they ask first
when a model is mid-answer and offer to stop it anyway. A wedged model has to be
stoppable — but by decision rather than by accident. Load never unloads
anything else: it is a manual act, for looking at one model, and if there is no
room it says what is in the way. On llama.cpp it loads anyway and lets the
engine leave the layers that will not fit in system memory.

**Settings** on a row shows what the model will be started with — context,
cache precision, how many requests at once, and per engine the rest. Changing
them means restarting the model, which is why the button says **Apply & reload**.
**Save** writes them down without touching the card.

## Library: what is on disk, and what could be

Every model found on disk, grouped by weight format, with what it can do and
whether the set is complete. Below it, a search of Hugging Face: pick a
repository, see which of its models this machine can actually run, and download
one whole — never file by file, because four shards of five is a model that
fails to load with an unhelpful message.

![The library](docs/screenshots/library.png)

A download goes to the folder for its format, decided by the server rather than
chosen. Deleting a model is refused while a configured entry points at it, and
that refusal sends you to Models to remove the entry first — two actions on two
pages, so it is always clear which kind of data is about to disappear.

## Settings: the machine, its engines, and where models live

Nothing on this page moves on its own. What is used, how warm the card is, how
many requests are running are facts about right now, and right now is the
Gateway page — a figure read here is still true an hour later.

![Settings](docs/screenshots/settings.png)

**Machine** is what this machine is, and the one thing that decides how many
models fit: how much of its memory is held back for the machine itself. A
dedicated card is used whole — nothing else here wants it. The machine's own
memory is shared with the browser and the operating system, so a reserve is
kept out of it. Set as a reserve rather than as an allowance, because a reserve
stays right when the machine is given more memory.

**Engines** shows what is installed, its version, and — only when there is one
— what it could be updated to. No button here updates anything: they open what
the update would bring, and the real Update is at the foot of that. See
[Updating an engine](#updating-an-engine).

**Paths** is every path this installation depends on: one models root with a
folder per weight format under it, and each engine's program. All of them are
picked from a listing of what is actually on the machine rather than typed — a
path typed by hand is a path with a typo in it, and the failure arrives much
later as a screen with no models on it.

## The Gateway: one address, and models that come and go

Each configured model is a separate engine on its own port. Pointed straight at
those, an agent naming a model that happens not to be running gets a refused
connection. The Gateway is one address in front of all of them, speaking the
OpenAI shape that agent tools already send:

```sh
curl http://ai-lab.lan:8090/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "reviewer", "messages": [{"role": "user", "content": "hello"}]}'
```

Name any configured model. No API key is checked; any value will do.
`GET /v1/models` lists every configured entry, loaded or not, which is the
point: a client is meant to be able to ask for one of them.

![The gateway](docs/screenshots/gateway.png)

The page is what is happening now. **Loaded** is how many models are on the
machine and the queue below names them. **Processing** is requests in flight
against the places every loaded model offers between them — places are the
engine's own number and differ per model. **Switches** counts loads and **Time
spent switching** is the share of the working time that went on loading rather
than answering: the number that says a workflow is thrashing.

The **Queue** is not a list of requests, it is what is about to happen. Each
loaded model with what it is answering and what is queued for it; **Waiting**,
the total queued for models already there; **Next change**, the model that has
to be loaded next; and **Remaining**, everything held up behind that change.

### How loading and unloading is decided

**Nothing is loaded or unloaded on a timer, or in the background, or because
something looked idle.** Every change happens because a request needs it.

**A request goes straight through only if all three hold:** nobody is waiting,
the model it names is loaded, and that model has a free place. Otherwise it
joins the back of the queue.

The first of those matters more than it looks. **The moment anybody is waiting,
the door closes for everybody** — including a request for a model that is
loaded and idle, which would cost nothing to serve. That is deliberate. Let
those past and the request at the head of the queue is never served, because
under a steady stream there is always another cheap one behind it. It is not
unlikely; it is certain.

**When a request finishes** and frees a place:

- the head of the queue wants a loaded model with room — it goes in, and again
  for as many as fit;
- the head wants something not loaded — nobody goes in. Room has to be made
  first;
- room can be made — the models chosen come off and the wanted one goes on.

**When the head of the queue wants a model that is not loaded**, how much room
it needs is worked out from the settings it asked for. If that fits in what is
free, it is simply loaded and **nothing comes off**. If it does not, models are
chosen to come off until it does:

1. **Idle models first**, longest-unused among them. Taking an idle model off
   costs no waiting.
2. **Then whichever answering model has gone longest without a request** — and
   everything stops until it has finished what it is doing. Cutting an answer
   short to make room is never done automatically; only the Unload button
   offers that, and it asks.

Because the door is shut while anybody waits, the models being drained cannot
pick up new work, so that wait is bounded by whatever was already in flight.

Nothing is protected from being unloaded. A model that is answering is waited
for, not spared.

**The run taken at a switch is the requests immediately behind the head that
want the same model, and no further.** The run stops at the first request
wanting something else, however many more of the first model are behind it.
Sweeping those up too would serve requests younger than one already waiting,
and the whole point of oldest-first is that a workflow may be held up by
exactly that older request. So the cost is visible rather than hidden: a
workflow alternating between two models that cannot both fit pays a load on
every step, and the fix is to reorder the workflow, not to change a setting
here.

**Requests arriving while a model loads wait for the next round**, even for the
model being loaded. Without that, the model just loaded starves the request
that was already waiting — the door problem again with the names swapped.

**How much a model needs is worked out each time and never remembered.** For
vLLM the answer is exact: it claims `gpu_memory_fraction` of the whole card and
the setting says which. For llama.cpp the weights are taken as a floor and
nothing more — measured at a 32k context, the gap between file size and card
usage ran from **−476 MiB to +6,663** across four models, so a computed cache
figure would look precise and be wrong. What a model took last time is
knowledge about the past, and it belongs to whatever is making the requests.

**A model that will not fit however much comes off is refused before anything
is disturbed**, with the numbers to correct by — see
[When a model cannot be loaded](#when-a-model-cannot-be-loaded). Unloading what
was working to load something that was never going to fit is the worst of both.

**A model that is running but that the Gateway did not load** — started from
the Models page, or left over from a manager restart — is taken up as it is
rather than reloaded. If it is still coming up it is left for later and looked
at again, because sending requests to a port with nothing behind it is worse
than waiting.

Two consequences worth designing around:

- A model plus the settings it was started with is one thing. Two requests for
  the same model wanting different context sizes cannot both be served without
  a reload.
- **A request must not wait, inside itself, on another request to this
  gateway.** Fill every place with things that cannot finish and nothing
  finishes.

### What it costs, measured

Two llama.cpp models loaded together on the RTX PRO 4500:

| | on the card | answered in |
|---|---|---|
| `gemma-general` | 3,544 MB | 0.04 s |
| `gemma26-gguf` | 20,539 MB together | 0.08 s |

Alternating between those two used to cost a load every time — 3.1 s and 7.3 s.
Neither leaves now.

Requests to one loaded model run together, up to the number the engine was
started to serve. Measured with Qwen3-Coder-30B at eight places: eight requests
at once took 1.2 s against 3.1 s one after another.

The gateway itself costs **1 ms** over talking to the engine directly.

### What fits beside what, measured

Every model here was loaded on the RTX PRO 4500 (32,623 MiB) and the card read
afterwards:

| model | engine | on the card |
|---|---|---|
| `gemma-4-E4B` | llama.cpp | 3,902 MiB |
| `gemma-4-26B-A4B` | llama.cpp | 18,184 |
| `Qwen3.6-35B-A3B` | llama.cpp | 21,880 |
| `gemma-4-31B` | llama.cpp | 24,614 |
| `qwopus27b` | vLLM | 27,382 |
| `glm47flash` | vLLM | 28,350 |
| `gemma-4-26B-A4B` | vLLM | 28,842 |
| `coder30b` | vLLM | 29,740 |

**A vLLM model takes a share of the whole card, not what its weights need.**
`gpu_memory_fraction` is 0.9 by default, and gemma-4-26b has 17.5 GB of weights
and occupies 28.8 GB — the rest is cache for concurrent requests. So two vLLM
models at the default never fit together, and lowering the fraction is the only
thing that changes that.

**llama.cpp takes roughly its weights plus a cache.** How much cache is not
worth predicting: at a 32k context the gap between file size and card usage ran
from **−476 MiB to +6,663** across four models, because architectures differ in
how they attend. So when the weights alone will not fit, llama.cpp is told to
put on as many layers as it can and leave the rest in system memory — it
measures the card at startup, which nothing outside it can do. That is slower:
moving 20 of 80 layers off the card cut prompt reading to a ninth here.
Generation suffers far less.

## Asking for a model a particular way

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

## Updating an engine

**Nothing updates without being read first.** The engine row has no button
that takes an update; it has one that shows what the update would bring, and
the real button is at the foot of that.

What it shows depends on how that engine arrives.

**llama.cpp is a git checkout**, so its changes are commits — already on disk
after a fetch, no service to be rate-limited by, and a great many of them. 138
were waiting on the container in one measurement, of which 26 were for Vulkan,
SYCL, Metal, OpenCL and ROCm, none of which exist there; 23 were build scripts
and 10 were its own web page. So they are sorted against what this machine
actually uses: 62 mattered, 76 did not. Nothing is thrown away — what does not
apply is counted and can be opened, because a summary that quietly drops things
is one nobody can trust.

Nothing about *this* machine is written down by hand. The accelerator decides
which hardware changes are worth reading, so the Mac wants Metal and the
container wants CUDA and neither is told which it is. The configured entries
decide the rest: the vision code only counts as yours if some entry points at a
model that can read pictures.

**llama.cpp has two release lines and you choose which to follow.** It tags
almost every commit to master as `b10448` — 7,160 of them in the checkout,
about seven a day. On 17 August 2026 it also began `v0.1.0` and up, in
upstream's own words *"stable, slower release cadence, recommended for
downstream distribution and casual users"*, with notes worth reading. Stable is
the default; `source.line` in `config.json` switches to `nightly`.

The two cannot be compared by their numbers — `b10448` and `v0.2.0` are not on
one scale — so "is there anything newer?" is asked of git instead: is that tag
already in this checkout's history? Exact, the same question on both lines, and
still right when you switch between them. An update moves to a named tag rather
than pulling whatever master has reached, so what got compiled has a name.

**vLLM arrives as ready-built packages.** Nothing is compiled and nothing is
recompiled indirectly. What there is instead is release notes written by
people, shown as they were written — sifting those by guessing at prefixes
would damage the one thing written for a reader.

But notes describe a *version* and say nothing about what happens to your
machine, so the package manager is asked what it would do, without doing it.
Measured on the container, moving from the installed nightly to 0.27.1:

| | |
|---|---|
| packages replaced | 15 |
| of those, **going backwards** | 8 |
| torch and the 2.9 GB of CUDA libraries | unchanged |

The eight backwards are `flashinfer`, `nvidia-cutlass-dsl` and its four
companion packages, `humming-kernels` and `quack-kernels` — because the
installed nightly had pulled newer kernel libraries than the stable release
pins. No release note anywhere says that, and it is the thing most likely to
break inference on the card.

### Installing beside what works, never over it

A new vLLM goes in a **new folder**, is checked that it actually imports, and
only then does the engine start using it. The previous one is untouched, so
going back is one press.

That is not caution for its own sake. The vLLM installed on the container could
not have been reinstalled: its wheel had left the local cache and the index it
came from — a nightly with a git hash in its name — is recorded nowhere on the
machine. An update in place would have been irreversible.

The engine is launched through a fixed path, a link called `current`, so
switching versions is repointing that link. It is done by writing a new link
beside it and renaming it over the top, in one step: a link deleted and then
recreated has a moment where it points at nothing, and anything starting in
that moment fails for a reason nobody would guess.

Settings lists the folders, marks the one in use, and offers the others as a
way back or as space to reclaim. **Nothing is ever tidied automatically** — the
previous version *is* the way back, and deciding it has stopped being needed is
a judgement about whether the new one has proved itself, which no timer can
make. Two folders is the steady state, about 16 GB.

One thing that cannot be moved: an environment created before this scheme
existed. A virtual environment's launcher scripts carry in their first line the
absolute path they were built at, so `/opt/ai/vllm/.venv/bin/vllm` starts
`/opt/ai/vllm/.venv/bin/python` **by name** — rename the folder and it starts
nothing. It is left where it is and recognised by its contents instead of by
its name. Everything installed since is created at its final path and has no
such problem.

Both kinds of update are refused while a model is loaded, and say which
entries to unload. The engine is about to be launched from somewhere else, and
a model already on the card would keep running the old one while the page said
otherwise.

## Structure

```text
ai_lab/hosts/           Platform: processes and accelerator, per operating system
ai_lab/engines/         One file per inference engine
ai_lab/downloads/       Hugging Face browsing and whole-set transfers
ai_lab/api/             HTTP routing and the progress event stream
ai_lab/web/             Browser interface, native ES modules, no build step
ai_lab/gateway.py       One address for an agent: routing by model name, and swapping
ai_lab/scheduler.py     Who gets the card next: the queue, the places, the swap
ai_lab/lastloaded.py    What was on the card, so a restart can put it back
ai_lab/catalog.py       Finding models on disk, grouping shards into sets
ai_lab/runtime.py       Load, unload and swap, with measured timings
ai_lab/settings.py      The settings view
ai_lab/capabilities.py  What a model can do, read from its own files and remembered
ai_lab/changes/         What an update would bring, read before anything is pressed
ai_lab/builds.py        Engine source versions, and updating them from git
ai_lab/installs.py      Engines that arrive as packages: versions side by side
ai_lab/operations.py    Joining the services into whole actions
ai_lab/config.py        Reading and writing config.json
ai_lab/naming.py        Rules about model file names
ai_lab/types.py         Shared data structures
ai_lab/events.py        Publishing progress to subscribers
ai_lab/wiring.py        Object construction only
ai_lab/main.py          Entry point: read the arguments, build, serve
tests/                  Unit tests, mirroring the package layout
system/                 systemd units and the one privileged helper
scripts/deploy.sh       Test and deploy to the AI-Lab container
docs/screenshots/       The interface, for the README
ARCHITECTURE.md         What each module is for, and what it must not contain
MODEL_STORAGE.md        Model store layout and NVIDIA weight formats
CLAUDE.md               Working rules for this project
opts/                   The private half: machines, credentials, real config
```

Read `ARCHITECTURE.md` before changing anything: it records the single
dependency direction the modules follow and the reason for it.

Runtime data is intentionally separate:

- application configuration: `/etc/ai-lab/config.json`, edited through the
  interface. The `config.json` in this repository seeds a fresh machine and is
  never copied over a running one, because instances created from the interface
  live in that file
- application state: `/var/lib/ai-lab`
- models: `/models`, a bind mount of the model volume on the internal disk,
  organised by weight format (see `MODEL_STORAGE.md`). One root is configured
  and each format is a folder in it, so the tree is the same on every machine
  and no format can be left behind on another disk
- deployed source: `/opt/ai-lab`

Generated state, virtual environments, logs and downloaded model weights stay
outside Git entirely.

## Deploying

`scripts/deploy.sh` tests locally, ships the tree, tests again on the far side
and restarts the manager. It has no address built into it — say which machine:

```sh
AI_LAB_HOST=root@proxmox.lan AI_LAB_CTID=102 ./scripts/deploy.sh
```

## Licence

MIT. See `LICENSE`.

## Local development

The application uses the Python standard library and supports Python 3.11 or newer.

```bash
python3 -m unittest discover -t . -s tests
python3 -m venv .venv
.venv/bin/pip install -e .
```

The tests need neither a GPU nor a network: the host and the engine are passed
in, so a fake stands in for both, and model directories are built from empty
files.

To run the manager on this machine:

```bash
./scripts/run-local.sh
```

On macOS it can instead run automatically after login, restart if it crashes,
and expose Open, Start, Stop and Restart from a menu bar icon:

```bash
./scripts/install-macos-service.sh
```

The installation is per user. It creates one Login Item for the small menu bar
app in `~/Applications`. That app owns the manager process, so Stop keeps it
stopped and a crash does not restart it. Logs are written to
`~/Library/Logs/AI-Lab`. To remove those installed files without touching the
configuration, logs or models:

```bash
./scripts/uninstall-macos-service.sh
```

It replaces any copy already running. There is no deployment step locally, so a
manager left running from before an edit keeps serving the old code, and the
symptom is confusing — the page looks wrong in a way the source says it cannot
be.

The full UI depends on Linux services, NVIDIA tooling and the model filesystem. Unit tests run locally on macOS; integration and GPU validation run in the LXC container.

## Deployment

`scripts/deploy.sh` performs the guarded deployment flow:

1. runs all tests locally;
2. streams the complete versioned project to the LXC runtime directory;
3. installs the versioned configuration, helpers and systemd units;
4. installs the package in the existing container environment;
5. runs the same tests inside the container;
6. restarts the web manager only after both test stages pass and checks that it is active.

Defaults target Proxmox host `mv` and LXC `102`. They can be overridden without editing the script:

```bash
AI_LAB_HOST=root@proxmox-host AI_LAB_CTID=102 scripts/deploy.sh
```

Downloaded models, mutable application state and active inference services are preserved during deployment.

## Safety model

The interface is intended for a trusted private network and does not implement application-level authentication. Operations that mutate state are guarded in the UI, while privileged service and model controls are restricted to narrowly scoped helpers.

## Origin

This project was designed iteratively as a human–AI collaboration: human intent, architecture, review and hardware validation combined with AI-assisted investigation and implementation.
