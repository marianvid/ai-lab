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

**It is still being built.** The current work is support for agentic workflows —
an agent naming a model that is not loaded, and getting it loaded rather than a
refused connection. It works; it is not written up here yet.

Provides a manager for local inference engines. It shows what models are on disk, what
is loaded on the accelerator right now, and how long a model took to load —
and it moves models on and off, measuring every step.

Runs on Linux with an NVIDIA card, where systemd supervises the engines, and
on macOS with Apple silicon, where it supervises them itself. llama.cpp works
on both. vLLM works on the Linux machine and is shown greyed out on the Mac,
with the reason — it needs CUDA, and Apple silicon offers Metal.

Read `ARCHITECTURE.md` before changing anything.

**How the models were measured lives in its own repository**,
[`ai-lab-benchmarks`](https://github.com/marianvid/ai-lab-benchmarks): four
tests across twenty languages on this machine, with the harness and the raw
results. It used to be a directory here; it is separate now so the application
stays an application.

## What it looks like

One line per configured model: what you called it, which model it runs, and —
at the right, against the buttons — the weight format and the engine that will
serve it. Port, context, temperature, state and the timings of the last load
live in the tooltip, because they are wanted occasionally and were costing three
lines of screen every time.

![The model list](docs/screenshots/models.png)

Settings reports the engines and the accelerator. llama.cpp is built from
source here, so its build number is shown and a newer one can be fetched and
compiled from this page. vLLM is a package rather than a source build, which is
why it has no version beside it and no update button.

![Settings](docs/screenshots/settings.png)

## Structure

```text
ai_lab/hosts/           Platform: processes and accelerator, per operating system
ai_lab/engines/         One file per inference engine
ai_lab/downloads/       Hugging Face browsing and whole-set transfers
ai_lab/api/             HTTP routing and the progress event stream
ai_lab/web/             Browser interface, native ES modules, no build step
ai_lab/catalog.py       Finding models on disk, grouping shards into sets
ai_lab/runtime.py       Load, unload and swap, with measured timings
ai_lab/settings.py      The settings view
ai_lab/builds.py        Engine source versions, and updating them from git
ai_lab/operations.py    Joining the services into whole actions
ai_lab/config.py        Reading and writing config.json
ai_lab/naming.py        Rules about model file names
ai_lab/types.py         Shared data structures
ai_lab/events.py        Publishing progress to subscribers
ai_lab/wiring.py        Object construction only
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
- models: `/models`, a bind mount of the model volume on the internal disk, organised by weight format (see `MODEL_STORAGE.md`)
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
