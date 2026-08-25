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

**One local address for text, speech and other AI models on a machine that
cannot hold them all.**

That is the whole point. A workflow may use one model to read, another to
write, one to transcribe a recording and another to detect speech. A 32 GB card
holds one of the large ones, or one large and one small. Pointed straight at the engines, a client naming a model that
happens not to be running gets a refused connection and the workflow stops
there.

AI-Lab is one address in front of all of them. A request names any configured
model; if it is loaded the request goes straight through, and if it is not,
**room is made and it is loaded first** — see [the Gateway](docs/gateway.md)
for exactly which model comes off and when. As many models stay loaded as the
memory allows, so the common case costs nothing: measured here, two models
served together answered in 0.04 and 0.08 seconds where alternating between
them used to cost a 3-second and a 7-second load.

**One thing is not standard and cannot be.** Some settings decide how a model
*process* starts — how much context it will hold, how much of the card it may
claim — and no chat API has a field for them. They travel in an `ai_lab` object
in the request body, which is specific to this project; a client that does
not send it gets the entry's configured settings and nothing breaks. See
[the `ai_lab` field](docs/requests.md#the-ai_lab-field-asking-for-a-model-started-a-particular-way).

Runs on Linux with an NVIDIA card, where systemd supervises the engines, and on
macOS with Apple silicon, where it supervises them itself. llama.cpp works on
both. vLLM, NeMo and the current speech services are Linux capabilities. An
unsupported engine stays visible but disabled, with the reason.

## Where things are

Each part has its own document. Every one of them says what the thing is for
before it says how it works.

**The five pages**

| | |
|---|---|
| [Models](docs/models.md) | One row per configured model: what it runs, what it can do, whether it is loaded and how long the last load took. Where models are started, stopped and set up. |
| [Library](docs/library.md) | What is on disk, per weight format, and a search of Hugging Face to download more. |
| [Gateway](docs/gateway.md) | The address an agent talks to. What is loaded, what is queued, and **the rules by which models are loaded and unloaded** — the part nobody can guess. |
| [Storage](docs/storage.md) | Cache, incomplete files and inactive engine versions whose space can be reclaimed. Model deletion stays in Library. |
| [Settings](docs/settings.md) | What this machine is, how much of its memory models may use, engine updates, and where the model store lives. |

**Using it**

| | |
|---|---|
| [Writing a request](docs/requests.md) | Chat and multipart audio requests, the `ai_lab` field for startup settings, and what a refusal contains so a client can correct itself. |
| [Updating an engine](docs/engines.md) | Reading what an update brings before taking it, and installing beside what already works so there is a way back. |
| [Audio](docs/audio.md) | Speech-to-text, VAD and speaker diarization, their endpoints and the input requirements specific to AI-Lab. |

**Working on it**

| | |
|---|---|
| [Working on it](docs/development.md) | What each module is for, running it locally, and deploying. |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | The one dependency rule every module follows, and the reason for it. **Read this before changing anything.** |
| [`MODEL_STORAGE.md`](MODEL_STORAGE.md) | How the model store is laid out on disk, and what each weight format is. |

## What it looks like

![The model list](docs/screenshots/models.png)

Two models loaded at once, each with what it runs, what it can do — a wrench
for tool calling, a photograph for reading pictures — the weight format, the
engine, and how long its last load took. More in [Models](docs/models.md).

![The gateway](docs/screenshots/gateway.png)

The address to point an agent at, what is loaded right now, and the queue read
as what is about to happen rather than as a list of requests. More in
[Gateway](docs/gateway.md).

## Licence

MIT. See `LICENSE`.

## Origin

This project was designed iteratively as a human–AI collaboration: human intent, architecture, review and hardware validation combined with AI-assisted investigation and implementation.
