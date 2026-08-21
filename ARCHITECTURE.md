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
| `runtime.py` | Load, unload and swap, with timings and progress events | Direct systemctl or nvidia-smi calls — it is handed a host |
| `downloads/` | Hugging Face browsing and fetching whole model sets | Deciding what a model *is* — that is the catalog's rule |
| `settings.py` | Assembling the settings screen from configuration and host | Writing to the accelerator |
| `builds.py` | Reporting an engine's source version, checking upstream, pulling and recompiling | Running engines, or choosing compile flags |
| `operations.py` | Joining the services into whole actions | Anything a single service could do alone |
| `gateway.py` | One address for an agent: which entry serves a name, and putting that model on the card | HTTP of any kind — forwarding is the web layer's job |
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

**One model on the card, one request at a time.** That is the design, not a
limitation being worked around. The models are chosen in advance and known to
fit; an agent workflow is a sequence, so there is nothing to gain from
overlapping requests and a great deal to lose from two arriving during a swap.
The consequence is worth stating plainly: **two agents on this machine do not
run in parallel.** The second waits for the first.

The rule has no exceptions. A request that needs no switch still unloads
anything else it finds running — a manager restart can leave two units up, and
a stray engine holds memory the loaded model wanted for context.

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

**The buttons on the page are the other way in.** Agent traffic is safe from
itself because every request takes the card in turn. Load, Unload and Apply
reach the engines directly and know nothing about who is mid-answer, so they
call `Gateway.guard` first. It refuses while the card is held, names the model
holding it, and the page offers to go ahead anyway — a wedged model has to be
stoppable, but that should be a decision rather than an accident. The refusal
travels as `409` with a `busy` object, because a page cannot act on a message
it would have to match on the words of.

### Why `naming.py` exists

The catalog applies shard rules to files on disk; the downloader applies the
same rules to a listing from Hugging Face. They must agree, or a model
downloads as one thing and appears in the library as another. Since neither may
import the other, the rules live below both.

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

**Asking what is running is the expensive question, and it is asked in bulk.**
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
