# Updating an engine

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

### Speech runtimes are isolated capabilities

NeMo ASR and the Silero ONNX adapter are installed in separate environments,
beside both the manager and vLLM. They follow the same operational rule — a
working environment is not modified indirectly by installing another engine —
but they do not yet have the automatic side-by-side update workflow described
above for vLLM.

The interface reports whether each runtime is available on the current host.
NeMo is offered for native `.nemo` transcription checkpoints. The ONNX adapter
is offered for VAD. On a host where the runtime or required accelerator is
missing, the engine is disabled with a reason rather than failing after an
instance has been configured.

Until managed updates exist, changing either speech environment is an explicit
deployment operation: record its package versions, validate import and device
access, run an inference smoke test, then deploy the manager. The versions used
for published measurements are recorded in the benchmark repository.

---

[← all documents](../README.md)  ·  [Audio](audio.md)  ·  [Writing a request](requests.md)  ·  [Working on it](development.md)
