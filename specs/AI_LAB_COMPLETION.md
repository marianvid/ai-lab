# AI-Lab completion specification

## Goal

Complete AI-Lab as the local inference service needed by Data-Lab and by the
planned image-production work. This phase installs no benchmark harness and
runs no model comparison. It exposes stable public operations, reports what
the machine can actually serve, and keeps model storage under operator control.

AI-Lab performs inference. It does not discover sources, demux video, rasterize
PDF pages, decide relevance, retain client products, or orchestrate a document
pipeline. Those remain Data-Lab or client responsibilities.

## Required operation matrix

| Operation | Public path | Engine | Consumer |
|---|---|---|---|
| Text generation | `/v1/chat/completions`, `/v1/completions` | llama.cpp or vLLM | Data-Lab and agents |
| Speech transcription | `/v1/audio/transcriptions` | NeMo or configured speech engine | Data-Lab |
| Voice activity detection | `/v1/audio/speech-segments` | ONNX/Silero | Data-Lab |
| Speaker diarization | `/v1/audio/diarizations` | Pyannote or NeMo | Data-Lab |
| OCR | `/v1/images/ocr` | self-hosted PaddleOCR/PaddleX | Data-Lab |
| Image generation | `/v1/images/generations` | self-hosted ComfyUI | image clients |
| Image editing | `/v1/images/edits` | self-hosted ComfyUI | image clients |

Video generation is not part of this phase. Audio extraction, media
normalization, subtitle discovery and timestamp reconciliation remain in
Data-Lab. Data-Lab sends only selected rasterized pages to OCR; AI-Lab does not
accept a whole PDF and decide which pages need OCR.

## Shared request rules

- Every request names an AI-Lab instance explicitly. No provider or model
  substitution is automatic.
- The `ai_lab` request field carries startup settings and is removed before an
  engine receives the request.
- Unsupported task, shape, model, workflow feature or memory requirement is a
  structured refusal made before loading weights.
- Load time, queue time, inference time, engine, model, effective settings and
  failure reason are returned as headers or structured metadata and emitted as
  operational events.
- Request bodies and generated content are not written to application logs.
- Engines remain reachable only through AI-Lab's documented gateway. Their
  private ports are implementation details.

## OCR contract

`POST /v1/images/ocr` accepts one image as multipart data plus:

- instance/model name;
- language hints;
- optional orientation and deskew controls;
- requested output detail: plain text, lines, or positioned blocks.

The response contains:

- full text;
- ordered lines or blocks with polygons/bounding boxes and confidence;
- detected language and orientation when available;
- model, engine, versions and effective settings;
- timing information and warnings.

The first engine is a local PaddleOCR/PaddleX service. It is managed through the
same engine interface as the existing engines and launched with an explicit
pipeline configuration. AI-Lab adapts the engine response to its stable OCR
schema; Data-Lab never depends on PaddleOCR's native response shape.

OCR acceptance requires Romanian and English fixtures, rotated text, a
low-quality scan, an image-only page and one mixed PDF workflow in which
Data-Lab calls OCR only for the page that needs it.

## Image generation and editing contract

ComfyUI runs as an AI-Lab engine, not as a second public product API. AI-Lab
uses ComfyUI's documented workflow API to upload inputs, submit work, observe
progress, cancel execution and retrieve outputs.

Clients do not submit arbitrary ComfyUI graphs. AI-Lab exposes named workflow
profiles stored in configuration. A profile declares:

- its versioned API-format workflow file;
- required checkpoints, encoders, VAEs, LoRAs and control models;
- supported operations: generation, edit, mask, references or controls;
- the input fields AI-Lab may replace;
- output nodes and expected media types;
- memory estimate and maximum dimensions.

`POST /v1/images/generations` accepts a prompt, named model/profile, dimensions,
count, seed and supported generation controls. `POST /v1/images/edits` also
accepts one or more reference images and an optional mask. Results include the
image bytes or opaque temporary result identifiers, exact seed, profile and
workflow versions, model component versions, effective settings and timings.

Image execution is represented as a pollable, cancelable AI-Lab inference job.
The synchronous OpenAI-compatible image routes may wait for that job, but the
native job state remains the recovery source of truth. Temporary ComfyUI inputs,
history and outputs are cleaned after delivery or expiry. AI-Lab does not make
long-term retention decisions for generated images.

Character-consistency scoring and the generated-image library belong to the
later benchmark phase, not this implementation.

## Capabilities and scheduling

Add `ocr`, `image-generation` and `image-edit` task values. Engines declare
their task, request paths, settings, concurrency and memory estimate through
the existing engine interface.

The public capability view must answer, per configured instance:

- whether it is available on this host;
- operations and request paths it serves;
- accepted input media and output forms;
- reference-image, mask, LoRA and control support where relevant;
- why an unavailable operation cannot run.

Text, audio, OCR and image requests share the accelerator through the current
oldest-first scheduler. A running image or OCR request is never interrupted by
a model switch. Cancellation removes queued work immediately and is cooperative
after engine execution starts.

Task-specific first-byte and total execution timeouts replace one timeout value
for every operation. A slow image workflow must not force text clients to wait
indefinitely, and a text timeout must not make valid image jobs fail.

## Core and benchmark model storage

AI-Lab exposes two configured model roots:

- `core`: the internal Lexar store used by production models;
- `benchmark`: `/test_models` on the external Corsair SSD.

The benchmark root is optional and may be disabled on a machine. New test
downloads default to it when enabled. Production downloads require an explicit
core selection.

Moving a model in either direction is a managed operation:

1. reject a model that is loaded, loading or referenced by an active job;
2. check destination identity, writability and free space;
3. copy to a temporary destination on the target filesystem;
4. verify the complete model set, file sizes and hashes;
5. atomically publish the destination;
6. remove the source only after successful verification;
7. retain the source and report a resumable failure if any earlier step fails.

Progress and errors are available through the normal event stream. The model
catalog scans both enabled roots and identifies each model's storage tier. A
ComfyUI `extra_model_paths` configuration maps both roots without duplicating
weights.

The Linux deployment adds `/mnt/corsair-4tb/test_models` on the Proxmox host as
the container's `/test_models`. The existing `/models` mount remains the core
root. No currently running media-copy job may be disturbed.

## Security and operational requirements

- Image and OCR uploads have configured byte, pixel and dimension limits.
- MIME type is verified from content rather than trusted from a filename.
- Workflow profile paths and output paths cannot escape configured roots.
- Arbitrary custom nodes and arbitrary workflow JSON are not accepted from
  remote clients.
- Temporary files use private directories and are deleted on success,
  cancellation, timeout and recovery after restart.
- Logs redact prompts, recognized text, image bytes, credentials and engine
  authorization headers.
- Health reports distinguish manager, engine process, model readiness and
  storage availability.
- Service restart reconstructs or explicitly fails unfinished inference and
  move jobs; it never reports an unknown job as completed.

## Implementation order

1. Finish and test model-root configuration and bidirectional managed moves.
2. Add task types, capability reporting and per-task timeout policy.
3. Add the PaddleOCR engine adapter and stable OCR route.
4. Add the ComfyUI engine adapter, named workflow profiles and inference jobs.
5. Add image generation/edit routes and temporary-result cleanup.
6. Update the non-technical UI for storage tiers, operations and live job state.
7. Deploy Linux mounts and services through documented management paths.
8. Run deterministic unit, API, security, restart and real-service smoke tests.
9. Update public documentation and private deployment documentation.

## AI-Lab completion gate

AI-Lab is complete for this phase only when:

- all existing text and audio tests still pass;
- every new route has deterministic success, refusal, cancellation and timeout
  coverage;
- no route silently falls back to another engine, model or provider;
- a real Linux OCR request succeeds through port 8090;
- a minimal real ComfyUI generation and edit request succeeds through port
  8090 without using ComfyUI's private port directly;
- core-to-benchmark and benchmark-to-core moves survive an injected copy
  failure without losing the source;
- restart recovery and temporary-file cleanup are demonstrated;
- the UI reports live state without exposing private engine paths;
- no benchmark result or private payload is committed.

## Data-Lab continuation gate

After AI-Lab passes its completion gate, resume the existing Data-Lab run and
implement the four currently blocking areas without changing approved scope:

1. media demux, normalization, subtitle extraction, ASR, VAD, diarization and
   timestamp reconciliation;
2. selective page OCR through AI-Lab;
3. atomic state-transition, event and webhook-outbox persistence;
4. callback ownership proofing and safe endpoint revalidation.

Data-Lab tests must use AI-Lab's documented port 8090 for real integration
checks. Engine-private ports, mocks that replace the integration gate, and
provider substitution are not acceptable completion evidence.

## Agent routing after approval

- Architecture ownership and adjudication: Codex frontier model in the
  controlling task.
- Implementation and documentation: Claude Sonnet.
- Deterministic tests and opposite-family code review: Codex Terra.
- Security review for uploads, paths, callbacks and temporary files: Claude
  Sonnet in a reviewer-only pass when Terra authored a fix; otherwise Terra.
- Holistic final review and release decision: Claude Opus.

An author never approves its own code. Failed evidence is preserved, and a
provider or model is changed only by an explicit recorded decision.
