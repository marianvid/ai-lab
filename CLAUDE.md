# Working rules for this project

## RULE 1 — WRITE LIKE THE "FOR DUMMIES" SERIES. THIS IS THE GOLDEN RULE.

**EXPLAIN THINGS IN PLAIN LANGUAGE. NO JARGON WITHOUT A PLAIN-WORDS EXPLANATION
RIGHT NEXT TO IT.**

Marian is a programmer, but nobody knows every field. Language that assumes a
specialist reader is language he has to decode instead of judge, and he cannot
review a decision he had to decode first.

**Calibrate correctly: he is an experienced programmer, not a beginner.**
Assume general programming literacy — HTTP, REST, classes, modules, processes,
threads, databases, git. Never explain those. What he does not have is depth in
every *specialty*: software-architecture vocabulary, LXC and container
internals, CUDA and quantisation formats, systemd details, GPU memory
management. Those are what need plain explanation.

The test: is this word general programming knowledge, or is it vocabulary from
one particular niche? Explain the second kind. Explaining the first kind is
condescending and wastes his time.

What this means in practice:

- Say what a thing **does** before saying what it is **called**. "The part that
  starts and stops the model process" comes before "the supervisor".
- When a technical term is genuinely the right word, define it once, in a short
  phrase, the first time it appears. Then use it freely.
- Prefer a concrete example over an abstract definition. "A model split into 5
  files plus its tokenizer is one thing, not six" beats "an aggregate root".
- Architecture jargon is the worst offender: *domain layer*, *transport*,
  *sideways import*, *composition root*, *aggregate*. If a term comes from a
  software-design book rather than from this project, replace it or explain it.
- Short sentences. One idea each.

This rule outranks brevity. A longer plain explanation beats a compact obscure
one every time.

## Rule 2 — Modular, never monolithic

One module, one job. Marian follows this project and intervenes in it, so he
must be able to open one file and understand one thing.

- No business decisions in the web-server layer.
- Modules import downward only. No service imports another service.
- Shared data types live in one shared file so modules never import each other
  just to borrow a type.
- One file per engine and per platform, behind a shared interface, rather than
  branching inside a common file.

Monolithic code has produced more bugs here. Treat a file that is accumulating
unrelated concerns as something to fix now, not later.

## Rule 3 — Act, do not ask for permission

Marian granted standing authorization for this project and for the `ai-lab`
container, local and remote. Execute and report at the end; do not interrupt him
to approve routine steps.

The one limit is system health. Still ask before:

- changing Proxmox networking without a recovery plan;
- detaching or resetting the NVIDIA PCI device while services run;
- hot-plugging OCuLink;
- deleting model files that are in use;
- disturbing the media-migration rsync jobs.

The separate `home-lab` LCD dashboard project is out of scope.

## Rule 4 — Verify, do not assume

Report what was measured, not what was expected. If something was not checked,
say so. Free disk space says nothing about whether a disk is available — the
internal `lexar-2` looks empty but is reserved for real-time data capture.

## Project facts worth knowing

- **ParallaxVox is Marian's company.** It belongs only as a branding label in
  the web page header — never in package names, paths, or unit names. The
  application itself is called AI-Lab.
- Model weights live on their own disk, organised by weight format. See
  `MODEL_STORAGE.md`.
- The deployed application runs in a Linux container on a Proxmox host. Which
  host, and how to reach it, is in the private half — `opts/MACHINES.md`.

## How to reach the machines

Addresses, credentials and console access are **not in this repository**. They
live in the private half, `opts/MACHINES.md`, together with the home-lab
document. See `opts/README.md` for why there are two repositories.
