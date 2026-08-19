# Architecture

Read this before changing anything. It says what each module is for and,
more usefully, what it must *not* contain.

## The one rule

Dependencies point in a single direction:

```
web  →  api  →  operations  →  services  →  engines  →  hosts
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
| `engines/` | Per engine: formats read, settings offered, command line built, readiness probe | Process supervision, filesystem scanning |
| `catalog.py` | Finding models on disk and grouping files into complete sets | HTTP, downloads |
| `runtime.py` | Load, unload and swap, with timings and progress events | Direct systemctl or nvidia-smi calls — it is handed a host |
| `downloads/` | Hugging Face browsing and fetching whole model sets | Deciding what a model *is* — that is the catalog's rule |
| `settings.py` | Assembling the settings screen from configuration and host | Writing to the accelerator |
| `builds.py` | Reporting an engine's source version, checking upstream, pulling and recompiling | Running engines, or choosing compile flags |
| `operations.py` | Joining the services into whole actions | Anything a single service could do alone |
| `api/` | HTTP routing, JSON, the event stream | Any decision about models, engines or formats |
| `web/` | The browser interface | — |

Four supporting files carry no policy of their own:

| File | Purpose |
|---|---|
| `types.py` | Shared data structures. No I/O of any kind. |
| `naming.py` | Rules about model file names — what a shard is, what a companion is. Pure text. |
| `events.py` | Publishing progress to subscribers. |
| `wiring.py` | Constructing objects and connecting them. No logic. |

### Why `operations.py` exists

Loading a model needs four things at once: the configuration says which
instance, the catalog finds the model on disk, the registry supplies the
engine, the runtime performs the move. None of them may import the others, and
the web layer must not make decisions — so the joining happens one layer above
the services and one below the routes.

The test for whether something belongs there: it reads as a sentence a user
would say. *Load this instance. Swap it to that model. Download this one.*

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

**A model either fits on the accelerator or it is not loaded.** There is no
partial-offload setting. Splitting layers between GPU and CPU sends every token
across the link, and on this machine that link is OCuLink, so the result would
be slow enough that nobody would want it. `gpu_layers` was removed for that
reason; a configuration that still names it has the key dropped rather than
rejected, so an older installation keeps working.

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
