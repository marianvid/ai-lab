# Refactoring plan

## Audit snapshot

The existing architecture has a useful dependency direction (`web → api →
gateway → operations → services → engines → hosts`), separate platform and
engine implementations, and substantial unit and browser coverage. Keep those
contracts while changing the internals. The principal maintenance costs are:

- `operations.py` (about 1,500 lines) owns unrelated use cases and is the
  shared dependency of most routes. `gateway.py` (about 1,200 lines) combines
  request admission, memory decisions, queue coordination and metrics.
- `runtime.py`, `builds.py` and `installs.py` each combine policy with process
  control. The browser's Models, Library and Settings views each exceed 490
  lines. Small changes therefore have a wide review surface.
- Configuration and API payloads move mostly as dictionaries. Validation and
  defaults are spread across constructors, services and views, so a new field
  can be accepted in one path and silently ignored in another.
- Platform defaults such as `/etc/ai-lab`, `/var/lib/ai-lab`, `/opt/ComfyUI`
  and port 8080 are embedded in code. Loopback addresses inside adapters are
  intentional, but installation paths and user-adjustable policy should enter
  through typed configuration. Avoid treating every literal as a defect.
- A catalog task such as `music-generation` is a classification, not proof
  that a runnable engine, request shape or browser workflow exists. Image
  profiles already have an API and a view, but the view was not registered.
- The public repository and private `opts` repository jointly define a
  deployable installation. They currently have no automated compatibility
  check. Never copy live paths or credentials into the public repository.

## Target package boundaries

Keep Python's `ai_lab` package and public HTTP contract. Introduce cohesive
subpackages as behavior moves, with one owner per state transition:

```text
ai_lab/domain/          Model, instance, task, job and resource value objects
ai_lab/application/     Use cases: instances, catalog, downloads, media, updates
ai_lab/ports/           Protocols for model store, config, host, engine, jobs
ai_lab/infrastructure/  File stores, OS supervision, HTTP clients, engine adapters
ai_lab/api/             Request parsing, response mapping and routing only
ai_lab/web/             Browser views and reusable controls
```

Each use case receives its dependencies in its constructor and returns typed
results. Use `dataclass` for immutable values and `Protocol` for boundaries;
retain engine classes as strategies. Keep side effects at adapters and express
errors as domain exceptions mapped to HTTP in one place. Avoid Java-style
getters, factories and interfaces where a plain value or small function is
clearer. New packages arrive as old modules are drained; do not move the
whole tree at once.

## Stages and acceptance gates

### 0. Establish a trustworthy baseline

Fix packaging so all subpackages enter the wheel, make the fresh-install seed
refer to the existing public example, and correct deployment documentation.
Gate: Python and browser suites pass; inspect a built wheel for audio,
changes and browser assets. This stage does not deploy.

### 1. Make existing capabilities usable

Give both model selectors a shared searchable control. Register Images as a
browser page. Add a task-aware direct-use link on each configured entry and a
small AI-Lab workbench for text, audio analysis and OCR. Keep llama.cpp's own
UI available. Gate: browser tests cover filtering, routing, requests and
navigation; existing image job tests remain green. A real ComfyUI generation
and edit on each configured machine are separate smoke checks.

### 2. Make configuration an explicit contract

Define typed configuration sections and a versioned migration path. Validate
engine/model/profile references, workflow files, paths, ports and task support
at startup with actionable errors. Add a read-only comparison command that
checks active Linux/macOS configuration against the versioned snapshots in
private `opts`, redacts sensitive values and reports drift. Gate: fixtures
from both platforms parse; round trips preserve unknown fields during the
migration; a deliberately invalid profile fails before a job is queued.

The configuration loader now preserves unknown top-level fields during a
save, and startup validates instance/engine/repository references, unique
ports, and image profile tasks and workflow files. The private verifier
compares active Mac and Linux bytes with `opts` snapshots after deployment.
Schema versioning and typed nested sections remain to be implemented.

### 3. Split application use cases

Move one vertical slice at a time out of `Operations`: instance lifecycle,
catalog/storage, downloads, engine updates, then media jobs. Routes call a
small application facade rather than the entire coordinator. Preserve endpoint
shapes while replacing dictionaries inside the application. Gate: contract
tests for each route, fake-host tests for load/unload and cancellation, and a
test that package imports work from an installed wheel. Delete old methods
after callers move; avoid duplicate policy.

Three extracted services now own configured instance lifecycle, model file
moves/deletion and model download selection. `Operations` remains the
API-facing facade until the remaining slices have moved.

### 4. Separate scheduling, resource admission and telemetry

Keep the current gateway behavior but give queue coordination, memory
admission, model leases and statistics separate owners. Assert invariants:
one lease is released exactly once; a queued model cannot starve; a cancelled
request never unloads a useful model; a busy model is not forcibly stopped
without the explicit path. Gate: deterministic concurrency tests with fake
clocks and hosts, plus Linux and macOS smoke tests against existing entries.

The eviction choice is now isolated in a pure planner. It prioritizes idle
models, refuses an impossible request before unloading anything, and has
focused tests. Gateway still owns memory readings, leases, scheduling
coordination and telemetry; those boundaries need separate follow-up work.

### 5. Add a first-class media studio

Give image and music jobs a common lifecycle contract (submit, inspect,
cancel, expire, download) with task-specific request and result types. Start
music with the already measured ACE-Step 1.5 XL Turbo on Linux and macOS;
wrap its isolated runtime behind an engine adapter and an AI-Lab job API.
Expose prompt, optional lyrics, duration, seed, progress and an in-browser
player/download. Add other music engines only after this one has a stable
contract. Keep a ComfyUI audio adapter possible for models such as MiniMax
Music 3 without forcing audio into the image response schema. Gate: fake
engine lifecycle tests, result persistence/cancellation tests and one real
generation on each platform, with the generated files kept private.

### 6. Shrink browser views and complete operational checks

Extract controls by behavior (model picker, settings editor, job list, result
viewer) and keep page modules responsible for composition only. Add accessible
labels, keyboard navigation and loading/error states. Run wheel, Python,
browser and configuration checks in CI. Deploy public code and private config
as separate changes, compare active state with private snapshots after each
deployment, and record the exact public/private commits that were verified
together. Gate: no private values in public diff, both repositories clean,
Mac and Linux smoke checks passed, and rollback instructions tested.

## Repository and deployment policy

`ai-lab` remains public application code, schemas, example configuration and
tests. `opts` remains private machine configuration, workflow profiles,
credentials and operational verification. Do not commit active configuration
from the UI blindly: compare and review the intended change first. A code
change that requires new private fields must include an example and validation
in the public repository and the corresponding snapshot change in `opts`
before deployment. Existing unrelated changes in the benchmark repository
must remain untouched.
