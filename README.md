# AI-Lab

> **Work in progress, built for personal use.** This is a home-lab tool that
> runs on one particular pair of machines, and it is still being changed
> frequently. It is public because the measurements and the approach may be
> useful to someone; it is not a product, has no support, and makes no promise
> of staying still.

**This was written with an AI agent, and it is meant to be read the same way.**
Take it as a starting point rather than as something to install. Your machine is
not this machine: different card, different amount of memory, different models,
a different idea of what the thing should do. Point your own agent at this
repository and have it adapt the code to what you have. That is a good deal
faster than reading it all yourself, and it is how the code got here.

**It is still being built.** Things move; expect the odd rough edge.

Provides a manager for local inference engines. It shows what models are on disk, what
is loaded on the accelerator right now, and how long a model took to load —
and it moves models on and off, measuring every step.

Runs on Linux with an NVIDIA card, where systemd supervises the engines, and
on macOS with Apple silicon, where it supervises them itself. llama.cpp works
on both. vLLM works on the Linux machine and is shown greyed out on the Mac,
with the reason — it needs CUDA, and Apple silicon offers Metal.

Read `ARCHITECTURE.md` before changing anything.

## What it looks like

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

Settings reports the engines, their versions and the accelerator. Both engines
can be updated from this page, and neither updates without being read first —
see **Updating an engine** below.

![Settings](docs/screenshots/settings.png)

> The screenshots are from an earlier state of the interface and do not show
> the capability icons or the version list.

## One address for an agent

An agent workflow uses several models — one to read, one to write, one to
check. Each is a separate entry here, on its own port, and only one of them can
be on the card at a time. Pointed straight at the engines, an agent naming a
model that happens not to be running gets a refused connection, and the
workflow stops there.

The Gateway is one address in front of all of them, speaking the OpenAI shape
that agent tools already send:

```sh
curl http://ai-lab.lan:8090/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "reviewer", "messages": [{"role": "user", "content": "hello"}]}'
```

Name any configured model. If it is the one on the card, the request goes
straight through. If it is not, the card is emptied and that model is loaded
first — the agent waits longer for that one request and sees nothing else. No
API key is checked; any value will do. `GET /v1/models` lists every configured
entry, loaded or not, which is the point: a client is meant to be able to ask
for one of them.

**One model on the card. Many requests to it.**

The card holds one model — that is the machine. Requests *to that model* run
together, up to the number the engine was started to serve: `parallel` for
llama.cpp, `max_sequences` for vLLM. That is what the engines are built for,
and it is the commonest shape agent traffic has — several subagents fanning out
over one model.

Measured here on an RTX PRO 4500 with Qwen3-Coder-30B at eight places: eight
requests at once took 1.2 s against 3.1 s one after another.

**Requests for a different model wait, and the queue is served in order.**
Requests next to each other wanting the same model go in together; the run
stops at the first one wanting something else. Nothing younger is served first,
even when it wants the model already loaded and would cost nothing — the fifty
requests for one model may be waiting on the answer to the one request for
another.

The cost is not hidden: requests that alternate between two models swap on
every one of them.

Two consequences worth designing around:

- A model plus the settings it was started with is one thing. Two requests for
  the same model wanting different context sizes cannot share a card.
- A request must not wait, inside itself, on another request to this gateway.
  Fill every place with things that cannot finish and nothing finishes.

The Gateway page reports what this is costing. The number to read is switches
as a share of requests: a workflow changing model on most of its steps spends
its time loading rather than working, and the fix is to reorder the workflow so
that steps sharing a model run together, not to change a setting here.

Load and Unload on the Models page reach the engines directly, so they ask
first when the card is mid-answer, and offer to stop it anyway. A wedged model
has to be stoppable — but by decision rather than by accident.

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

### Asking for the model started a particular way

Some settings go in a request — temperature, top-p, how many tokens to write.
Others decide how the model *process* starts, and cannot: context size is the
one that matters, because a model has to be told at startup how much it will
be asked to hold.

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

Nothing here interrupts an answer in progress. Every request takes the card in
turn and holds it to the last byte, so a request wanting different settings
waits like any other. That is why the Load and Unload buttons need a guard and
this does not — the buttons do not queue.

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
