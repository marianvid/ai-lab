# Architecture

Read this before changing anything. It says what each module is for and,
more usefully, what it must *not* contain.

## The one rule

Dependencies point in a single direction:

```
web  →  api  →  gateway  →  operations  →  services  →  engines  →  hosts
```

Nothing imports upward, and **no service imports another service**.

The reason is practical rather than doctrinal. When two modules depend on each
other, a change to either can break the other in a way nothing catches until
runtime. With one direction, a break is always downstream of the edit, so
there is exactly one place to look.

## Modules

| Module | Owns | Must not contain |
|---|---|---|
| `hosts/` | Starting and stopping processes, reading the accelerator, saying what this machine supports | Anything about models or engines |
| `hosts/base.py` | The questions every platform must answer, including `statuses` — the same question as `status`, asked about several instances at once | — |
| `engines/` | Per engine: formats read, settings offered, command line built, readiness probe | Process supervision, filesystem scanning |
| `catalog.py` | Finding models on disk and grouping files into complete sets | HTTP, downloads |
| `capabilities.py` | Reading a model's own files to find out whether it can call tools or read pictures, and remembering the answer | Which engine will run it, and what any setting says |
| `runtime.py` | Load, unload and swap, with timings and progress events | Direct systemctl or nvidia-smi calls — it is handed a host |
| `downloads/` | Hugging Face browsing and fetching whole model sets | Deciding what a model *is* — that is the catalog's rule |
| `settings.py` | Assembling the settings screen from configuration and host | Writing to the accelerator |
| `builds.py` | Reporting an engine's source version, checking upstream, moving to a tag and recompiling | Running engines, or choosing compile flags |
| `changes/` | What an update would bring, read before anything is pressed: commits waiting, notes written upstream, packages that would be replaced | Doing the update — it only reads |
| `installs.py` | The installed versions of an engine that arrives as packages: adding one beside the others, choosing between them, dropping one | Compiling anything, or deciding when an old version stops being needed |
| `operations.py` | Joining the services into whole actions | Anything a single service could do alone |
| `gateway.py` | One address for an agent: which entry serves a name, and putting that model on the card | HTTP of any kind — forwarding is the web layer's job |
| `scheduler.py` | Who gets the card next: the queue, the places, the decision to swap | Anything about models, engines or ports — a shape is an opaque key |
| `lastloaded.py` | One fact on disk: which model was on the card and how it was started | Deciding anything — it remembers and is read |
| `api/` | HTTP routing, JSON, the event stream | Any decision about models, engines or formats |
| `web/` | The browser interface | — |

Four supporting files carry no policy of their own:

| File | Purpose |
|---|---|
| `types.py` | Shared data structures. No I/O of any kind. |
| `naming.py` | Rules about model file names — what a shard is, what a companion is. Pure text. |
| `events.py` | Publishing progress to subscribers. |
| `wiring.py` | Constructing objects and connecting them. No logic. |
| `main.py` | Reading the arguments, building the application, serving. Stops the engines on the way out, where this application is the one supervising them. |

### Why `operations.py` exists

Loading a model needs four things at once: the configuration says which
instance, the catalog finds the model on disk, the registry supplies the
engine, the runtime performs the move. None of them may import the others, and
the web layer must not make decisions — so the joining happens one layer above
the services and one below the routes.

The test for whether something belongs there: it reads as a sentence a user
would say. *Load this instance. Swap it to that model. Download this one.*

### Why `gateway.py` exists

An agent workflow uses several models — one to read, one to write, one to
check. Each is a separate entry on its own port, and only one of them can be on
the card. Without something in front of them, an agent naming a model that is
not running gets a refused connection and the workflow stops there.

The gateway is that front. A request names a model; if it is loaded the request
goes through, and if it is not, the card is emptied and that model is loaded
first. The agent waits longer for that one request and sees nothing else.

It sits above `operations.py` because it is a policy about *which* whole action
to perform, not a whole action itself. It sits below `api/` because it contains
no HTTP: it decides which entry serves a name and makes sure that entry is the
one on the card. Forwarding the request and streaming the answer back are the
web layer's job.

**One model on the card. Many requests to it.** The card holds one model —
that is the machine. Requests *to that model* run together, up to the number
the engine was started to serve, because that is what the engines are built
for: vLLM interleaves them in one pass, measured at up to seventeen times the
throughput as concurrency rises against about 1.4 for llama.cpp. Making them
take turns threw that away.

The number is the engine's own, per entry — slots for llama.cpp, sequences for
vLLM — and `Engine.concurrency` is where it is asked for. Guessing from a
setting name would mean this layer keeping a list of names it has to be told
about.

It used to be one request at a time, and the reasoning written here was that an
agent workflow is a sequence. That stopped being true the moment subagents
fanned out over one model, which is the commonest shape there is. The premise
changed; the conclusion had to.

**Requests for a different model queue, and the rules are in `scheduler.py`.**
Two of them are easy to get wrong and impossible to notice afterwards:

- The moment anybody waits for another model, the door closes for the model on
  the card. Without it a busy model never goes idle, the switch never happens,
  and the other request waits for ever. Under continuous load that is not
  unlikely, it is certain.
- What arrives while a model is loading waits for the next round, even if it
  wants the model being loaded. Without it the model just loaded starves the
  one that was waiting — the same fault with the names swapped.

**The queue is served in order, and requests next to each other wanting the
same model go in together.** That is the whole rule. The run stops at the first
request wanting something else, however many more of the first model are behind
it.

Sweeping those later ones up is tempting — same model, already loaded, free —
and it was written that way first. They arrived after the request that wants
something else, though, and serving them ahead of it is what oldest-first
exists to prevent. A workflow can be held up by that older request, and no
number of cheaply-served younger ones makes up for holding it longer.

The cost is not hidden: requests that alternate between two models swap on
every one of them. Two models genuinely needed at once is the card's limit, and
no ordering rule escapes it. Nothing is traded for fewer loads — no dwell time,
no batching by size — because the fifty requests for one model may be waiting
on the answer to the one request for another.

Which gives the one thing to design workflows around: **a request must not
wait, inside itself, on another request to this gateway.** Fill every place
with things that cannot finish and nothing finishes. Fairness does not save you
from that; nothing does.

**A model plus the settings it was started with is one shape.** Two requests
for the same entry wanting different context sizes are not requests for the
same thing — one of them needs a reload — so the settings are part of what is
being asked for rather than a note attached to it.

**A client that gave up is dropped when its turn comes**, before anything is
unloaded. Reproduced on the machine before this existed: a client asked for
another model and hung up at once, and the manager took a working model off the
card to load twenty-one gigabytes for nobody. Checking after the load reproduces
it exactly, so the check is part of taking the photograph.

**The lock is never held while a model loads.** That takes up to a minute, and
everything would stop for it — including the page asking what is going on,
which is exactly when somebody wants to know.

**A switch empties the card and waits before it loads.** The driver returns
video memory a moment after a process exits, and starting a model on top of
memory that has not come back yet fails in a way that reads like the new model
being too large — which sends whoever reads it looking in the wrong place. So a
switch unloads, polls until the card is under 512 MB, and only then loads. If
it never goes quiet, the switch fails and says what is still holding it. On
unified memory there is nothing to wait for and the step is skipped.

**A request arrives in one of two shapes, and the engine says which it reads.**
Nearly every client speaks the OpenAI shape. A client written against
Anthropic's own library posts to `/v1/messages` instead — the same models
answering, only the wording of the request differs. vLLM serves both; llama.cpp
serves one.

`Engine.api_paths` is where that is stated, next to the formats the engine
reads and the settings it takes, so nothing above has to know one engine from
another. The front door registers every shape any engine can answer, and an
entry asked for a shape its engine cannot answer is refused before the card is
touched — with the entries that *would* have worked, because a client does not
know which of its models runs on which engine and a refusal that only says no
leaves it nowhere to go.

Refusing early matters: the alternative is a forty-second load followed by a
404 from the engine about a path the client never chose.

**An entry has one name, and it is typed rather than derived.** There used to
be two: an id that requests carried, and a label a person read. The label was
what the page showed and the id was what worked, so the name to send appeared
nowhere in the interface and had to be guessed or read out of the
configuration file. Worse, the gateway answered to four names for one entry —
id, label, model path, and the file at the end of it — and the first match won.

Now the id is the whole of it. It is given when the entry is made, checked for
being unique and for holding only lower-case letters, digits and hyphens, and
cannot be changed afterwards: a request carries it, so renaming would break
whatever is already sending it. Renaming is deleting and adding, which the page
already offers.

The rules are shown where the name is typed rather than in the message that
refuses it, and a configuration written before this loads unchanged with the
label dropped.

**A request can ask for the model started a particular way.** Some settings go
in a request; others decide how the process starts and cannot. Context size is
the one that matters — a model is told at startup how much it will hold, and an
agent needs room for its own instructions. Sent in the body it would reach the
engine, which does not know it and ignores it silently, so it travels in a
field of its own that the web layer reads and removes before forwarding. What
reaches the engine is exactly what would have reached it before.

**They are not saved.** `Operations.load` takes them and lays them over the
entry's own without writing anything: one request must not rewrite what
somebody chose in the page. The running model then differs from its
configuration, so `Runtime` remembers what it actually launched with and
reports the difference — otherwise the page shows a number the running model is
not using and nothing says so.

The comparison is against what is *running*, not what is configured. An earlier
request may have already reloaded it with exactly these settings, and reloading
again would cost the wait for the same answer.

Nothing here interrupts an answer. The card is taken first and only handed over
after the request in front has had its last byte, so a request wanting
different settings queues like any other. That is the difference between this
and the buttons below, which do not queue and therefore need `guard`.

**The buttons on the page are the other way in.** Requests here are safe from
each other because they queue. Load, Unload and Apply reach the engines
directly and know nothing about it, so they call `Gateway.guard` first. It
refuses while anything is running or waiting and says what it found — one
answer being written is a different thing from forty requests waiting for a
model, and the difference decides whether to go ahead.

Going ahead means a clean slate: everything in flight dies, everyone waiting is
turned away, the card is empty. Half-forced is worse than either, because the
queue behind a stopped engine describes a world that will not exist a second
later. The routes also tell the gateway when a button moves a model at all,
since the scheduler would otherwise admit a request onto a card holding
something else.

The refusal travels as `409` with a `busy` object, because a page cannot act on
a message it would have to match on the words of.

### Why `naming.py` exists

The catalog applies shard rules to files on disk; the downloader applies the
same rules to a listing from Hugging Face. They must agree, or a model
downloads as one thing and appears in the library as another. Since neither may
import the other, the rules live below both.

### Why `capabilities.py` exists

Two things a model can do are worth seeing before choosing it: whether it can
call tools, and whether it can read pictures. Neither is written down anywhere
in this project's configuration — both are properties of the weights, and the
only honest place to learn them is the model's own files.

For a directory of weights it is the config beside them, and pictures need two
marks rather than one: a vision section *and* a token to put a picture in. A
text model's config can name a vision tower it does not use, and one mark alone
was enough to claim a model could see when it could not.

For GGUF it is the chat template, which sits inside the single weights file
about 6 to 16 MB in, behind the vocabulary. A template that mentions tools is a
model that was taught to ask for them. Pictures there are a separate file
beside it, `mmproj-*.gguf`, which is what llama.cpp is handed to see with — and
so a copy of the same model downloaded without that file genuinely cannot see,
which is what the library then shows.

Reading it costs about a quarter of a second per GGUF model: 2.1 seconds for
the twenty models on the container, measured, and that is on the way to drawing
a page. So the answer is written down, in the state directory beside everything
else the manager remembers, keyed by the path with the file's size and
modification time. The second read of that library takes 26 ms. A model
replaced under the same name is read again rather than believed. A finished
download reads the new model straight away, on the download worker, so nobody
waits for it in front of a page.

**The weights decide what a model can do; a setting can only take something
away.** vLLM's "Text only" loads a model that can see without the part that
sees. So the interface subtracts: capabilities come from the files, the entry's
settings remove what they switch off, and what is left is what the running
model will actually do. Nothing may add a capability the weights do not have.

### Why `changes/` exists

An update should be a decision, not a hope. Before either engine is updated,
this answers *what would change?* — and the two engines are genuinely
different, so they are not forced to look alike.

llama.cpp is a checkout, so its changes are commits: already on disk after a
fetch, no network beyond that, and a great many of them. 138 were waiting on
the container in one measurement, of which 26 were for Vulkan, SYCL, Metal,
OpenCL and ROCm — none of which exist there — 23 were build scripts and 10 were
its own web page. So they are sorted against what this machine actually uses:
62 mattered, 76 did not. Nothing is dropped; what does not apply is kept behind
a count, because a summary that quietly throws things away is one nobody can
trust. Nothing about *this* machine is written down by hand — the accelerator
decides which hardware matters, and the configured entries decide the rest,
including whether the vision code is worth reading at all.

vLLM arrives as ready-built packages, so there are no commits and nothing is
recompiled. What there is instead is better — notes written by people, for
somebody about to install — and those are shown as written. Sifting a release
note by guessing at prefixes would damage the one thing that was written for a
reader.

But notes describe a *version*, and say nothing about what happens to this
particular machine. So `uv` is asked what it would do, without doing it.
Measured on the container: moving from the installed nightly to 0.27.1 replaces
fifteen packages and takes **eight of them backwards**, because the nightly had
pulled newer kernel libraries than the stable release pins. No release note
anywhere says that, and it is the thing most likely to break the card.

**llama.cpp has two release lines, and which one to follow is configuration.**
It tags almost every commit to master as `b10448` — 7,160 of them in the
checkout, about seven a day. On 17 August 2026 it also began `v0.1.0` and up,
in upstream's own words "stable, slower release cadence, recommended for
downstream distribution and casual users", with notes worth reading. The
default is stable. The two cannot be compared by their numbers — `b10448` and
`v0.2.0` are not on one scale — so "is there anything newer?" is asked of git:
is that tag already in this history? Exact, the same question on both lines,
and still right when somebody switches between them.

### Why `installs.py` exists

vLLM is not compiled here — it is 382 packages and 7.7 GB in a virtual
environment, a folder holding its own Python and everything that version needs.
Two of them sit side by side without touching each other, and that is the whole
design: **a new version goes in a new folder, and what works is never written
over.**

That is not caution for its own sake. The vLLM installed on the container when
this was written could not have been reinstalled: its wheel had left the local
cache, and the index it came from — a nightly with a git hash in its name — is
recorded nowhere on the machine. An update in place would have been
irreversible, and if the new one had not supported this card there would have
been nothing to go back to.

So: build the new one, check that it actually imports, and only then point the
engine at it. The engine is launched through a fixed path — a link called
`current` — so switching versions is repointing that link, done by writing a
new link beside it and renaming it over the top. A link deleted and recreated
has a moment where it points at nothing, and anything starting in that moment
fails for a reason nobody would guess.

Two folders is the steady state, about 16 GB. It does not grow on its own and
**nothing is ever tidied automatically**: the previous version is the way back,
and deciding it has stopped being needed is a judgement about whether the new
one has proved itself. A timer cannot make that judgement.

One thing that cannot be moved: the environment that predates this scheme. A
virtual environment's launcher scripts carry the absolute path they were built
at in their first line, so `/opt/ai/vllm/.venv/bin/vllm` starts
`/opt/ai/vllm/.venv/bin/python` **by name** — rename the folder and it starts
nothing. It is therefore left exactly where it is and understood by its
contents rather than by its name. Everything installed since is created at its
final path and has no such problem. A copy taken as a precaution before this
existed had a first line pointing back at the original and would not have
worked without it; that is measured, not supposed.

## Two decisions worth knowing

**Processes are supervised differently per platform.** On Linux, systemd owns
the engine processes, so inference survives a manager restart and returns after
a reboot; the unprivileged manager reaches systemd through one narrow sudo
helper. On macOS this application starts the processes itself, so inference
stops when the manager stops. Both sit behind `hosts.Host`, and the difference
is visible to the interface through `Capabilities` — never through an operating
system check in the code.

Instances are created at runtime, so the command line cannot live in a static
unit file. The manager writes it to `/var/lib/ai-lab/launch/<id>.json` and a
templated unit runs `ai-lab-run <id>`, which reads the file and execs. That
launcher understands nothing else. The unit runs as `ai-lab-manager`, the same
user as the manager, so a manager-written command line grants no privilege the
manager did not already have.

**Asking what is running is the expensive question, and most work does not
need it.** Which entry answers to a name, which engine runs it, what settings
it has — that is the configuration file, 0.05 ms to read. What every instance
is *doing* costs 73 ms, and the front door needs it only when something outside
may have moved a model: at startup, or after a button on the page. `configured`
answers the first kind of question and `instances` the second, and the
difference between them is the whole of the gateway's per-request cost —
measured on the container at 19 ms through it against 17.8 ms straight to the
engine.

Two things were paying for it without needing to. Working out an entry's
effective settings went through `_resolve`, which also finds the model on disk
and so walked every model directory: 11 ms for an answer made entirely of
configuration and the engine's own rules. And the front door asked what
everything was doing on every request, to notice a card somebody else had
changed — which it now asks once, after being told it might have been.

**Asking what is running is done in bulk when it is done at all.**
Drawing the model list needs the state of every configured instance, and the
gateway needs it on every request. On systemd each answer is a command — three
per instance, for whether it is active, whether it is enabled, and its pid — so
with eleven instances configured that reading cost 152 ms, which was the entire
cost of the call; the readiness probes beside it were free. A one-token request
that the engine answered in 17 ms was taking 500 ms to reach it.

`Host.statuses` asks about every instance at once. `systemctl show` accepts as
many units as it is given and answers with one block each, so the same
information costs one command rather than thirty-three: 152 ms became 15, and
the request overhead 500 ms became 140.

The method is on the interface with the one-at-a-time loop as its meaning, and
only the Linux host overrides it. On macOS this application owns the processes
and keeps them in a dictionary, so asking about one is a lookup and there is
nothing to batch — measured there, the gateway costs 39 ms against 40 ms
straight to the engine. A platform with nothing to gain says so and moves on.

**What is on the card survives a restart, and nothing else decides it.**
There used to be no answer to "what comes up after a reboot". The unit files
allow an engine to be enabled, and two of them had been, by hand, months
earlier — so a machine came back with two models on a card that holds one, and
nothing in the application could say otherwise. That flag is gone from the
interface, from the status a host reports, and from the units.

In its place, the model on the card is written down as it changes and put back
when the manager starts. Three cases, and the second and third are the ones
worth stating:

- **A machine rebooted.** Nothing is running, something was remembered, so it
  is loaded again — with the settings it was actually started with, since a
  request can ask for a bigger context than the entry is configured for and
  bringing back the same model set up differently would be a poor restore.
- **A manager restarted.** On Linux systemd owns the engines and they survive,
  which is the reason for using it. The manager comes back, finds its model
  still answering, and leaves it alone. Reloading would take the card away
  from whoever is using it.
- **The card was emptied on purpose.** That is remembered as firmly as a model
  is. Somebody who unloads and then reboots does not want it back.

The restore runs in the background: a large model takes the better part of a
minute, and the page should open at once showing it loading rather than refuse
to open until it has. A restore that cannot work is reported and dropped — this
runs while the manager is starting, and a model whose files have gone must not
stop the manager from serving.

Unloading names what it is forgetting. The gateway unloads a stray from beside
the model that stays, and that must not read as the card having been emptied.

**Engines outlive the manager on Linux and not on macOS, on purpose.** systemd
owns them there, so restarting or deploying the manager does not interrupt
inference — the point of using it. On macOS this application is the
supervisor, so it stops its engines when it exits, including on SIGTERM, which
Python's exit handlers do not cover. An engine left holding a port would
otherwise answer the next manager's health probe and turn a load that never
happened into a reported success. `Host.stop_all` is the seam: real on macOS,
deliberately empty on Linux.

A load also refuses to start when something is already answering the port that
this manager did not start, for the same reason.

**Updating an engine rebuilds; it does not reconfigure.** Both machines
compile llama.cpp from git, and the existing `build/` directory already holds
the flags each was set up with — CUDA compiled for this exact card, Metal with
embedded shaders. `cmake --build` reuses them. Regenerating the configuration
would mean guessing those flags, and guessing wrong is silent: a working binary
that quietly lost an optimisation.

An update is refused while any instance is running, because a linker cannot
write over an executing binary — the build would fail partway with a message
about a busy file instead of a sentence saying to unload the models. And an
update is not offered when the installed version cannot be read: a shallow
clone has no tags, its version compares as zero, and every remote tag would
look newer.

**Engines describe themselves.** An engine declares its settings through
`ParamSpec`, and that one declaration both validates incoming values and draws
the form. The path to its binary comes from configuration rather than PATH: a
machine can easily hold two builds of llama.cpp — one packaged, one compiled
with the flags you wanted — and PATH order is nobody's decision. This is why the configuration nests engine settings under `params`
rather than listing every engine's fields side by side, and why adding an
engine touches neither the schema nor the front end.

**How long to wait for an engine: two limits, both of safety.** In normal work
nothing comes near them.

*To the first byte* covers connecting and reading the prompt — and, for a
request that did not ask for streaming, the whole answer, because such an
engine sends nothing until it has finished. *Between bytes* catches an engine
that starts answering and stops.

One number cannot do both jobs. At the slowest generation measured on either
machine, 17 tokens a second, the gap between tokens is 59 milliseconds, while a
large prompt on that same machine can take minutes before the first byte. Four
orders of magnitude apart. There used to be one limit, of an hour, which was
absurd for one job and useless for the other.

They and the queue length are configured rather than sent with a request: they
belong to the machine, and the right numbers on a card that reads 8,400 tokens
of prompt in under a second are not the right numbers on a Mac running a 70 GB
model at 17 tokens a second. They are edited on the Gateway page, beside the
figures somebody would read before deciding to change them.

## Measuring a load

Each step is timed:

```
unload:  stopping → process_gone → memory_released
load:    starting → process_up → weights_loading → ready
```

`process_up` means the binary is running. `ready` means it answered its own
health probe, which is the only trustworthy sign that the weights are
resident. The interval between the two is the interesting one.

During a transition the accelerator is sampled about five times a second and
every reading is published as an event. The browser draws those readings
directly, so the bar follows memory actually rising or falling rather than a
timed animation. Measured on an M3 Max, loading a 400 MB model:

```
starting          11 ms      0.0 MB
process_up        36 ms      8.0 MB
weights_loading  259 ms     92.1 MB
weights_loading  476 ms    598.4 MB
ready            700 ms    604.4 MB
```

An unload waits for memory to stop falling, not merely for the process to
exit: the two are different moments, because the driver frees asynchronously.

**The bar is the operation's progress, not the card's occupancy.** Each event
carries a `progress` between 0 and 1, worked out by the runtime, and the
browser simply draws it. Occupancy makes a poor bar: a 4 GB model on a 32 GB
card would sit at 13% while being completely loaded, and an unload would end
wherever the *other* resident model happened to leave it.

The long phase has a known destination — the weights have to arrive in full —
so it is reported as a real fraction of the model size rather than a guess. The
short phases either side get small fixed slices, because a bar that sits at
zero and then jumps is worse than one that moves while a process starts. A swap
is one continuous bar: the unload takes the first 40%, the load the rest.

Memory is still reported, as numbers rather than as the bar, and per instance
rather than per card. On Linux that figure comes from `nvidia-smi
--query-compute-apps`, whose pids inside the container match the ones systemd
reports; on macOS it is the process's resident memory.

**A model that does not fit is refused, unless the split was asked for.**
Before anything starts, the weights are compared against the free memory on the
card. That is a necessary condition and not a sufficient one — the context
cache sits on top of the weights — but when the weights alone exceed what is
free there is no point starting a process to find out, and the alternative is a
confusing crash a few seconds later.

`gpu_layers` says how much of the model goes on the card. It was removed once,
on the grounds that splitting a model between card and system memory sends
every token across the link — OCuLink here — and is slow enough that nobody
would want it. It came back as a setting rather than a rule, because how slow
is a measurement and whether it is worth it is not the code's decision. The
setting says what it costs, in the words a person reads before choosing it:
moving 20 of 80 layers off the card cut prompt reading to a ninth on this
machine, while generation suffered far less. A plan that deliberately leaves part of the model
in system memory is exempt from the fit check: there the weights are not all
meant to be on the card, so comparing the whole file against free memory
answers a question nobody asked.

On Apple silicon there is no separate video memory, so the readings carry
`memory_kind: "unified"` and report the engine process's resident memory
instead. The bar still works; it measures a different quantity, and says so.

## Testing

Everything below the service layer is injected, never imported, so tests
supply a fake host and a fake engine and need no GPU. Filesystem tests build a
directory of empty files, because a model is defined by its names and sizes.

One rule earns its keep repeatedly: an engine's readiness probe reaches the
network, so tests pass an engine whose probe answers immediately. Every other
rule — formats, validation, command line — stays the real one.

```
python3 -m unittest discover -t . -s tests
```
