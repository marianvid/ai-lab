# Direct model interfaces

AI-LAB manages a model installation, an inference runtime and a human interface
as separate things. An interface is a client of a runtime; it is not another
copy of the model. An agent and a person must reach the same configured model
through the same capacity and lifecycle rules.

## Four objects

| Object | Owns | Identity |
|---|---|---|
| `ModelInstallation` | A checkpoint on one host, task, engine selection and saved launch settings | Host + configured instance ID |
| `EngineRuntime` | Starting, probing and stopping inference; supported request shapes and memory estimate | Existing engine strategy + runtime instance |
| `InterfaceProvider` | Compatibility, install/probe/start/open behavior, supported controls and traffic mode | Provider ID and version |
| `InterfaceDeployment` | One installed UI service on one host; may serve several installations and browser sessions | Host + provider + deployment ID |

`DirectSession` binds a user, installation and interface deployment while the
page is open. The session owns a reservation only where the UI sends directly
to a model process. Browser tabs sharing one deployment do not launch extra
model copies. A model may be offered through more than one compatible UI; a
single UI may list several configured models. The preferred binding is
configuration, with an optional per-instance override, not a checkpoint name
embedded in Python or JavaScript.

`InterfaceProvider` is a small strategy contract: `supports(installation,
host)`, `probe(deployment)`, `start(deployment)`, `open(session)` and
`controls(installation)`. Implementations may wrap an engine's embedded page,
launch a separate UI service, or expose a workflow application. Process
supervision stays in `hosts/`; request admission stays in `gateway/`.

## Package boundaries

```text
ai_lab/interfaces/types.py       Provider, deployment, binding and session values
ai_lab/interfaces/providers/      One adapter per maintained UI family
ai_lab/interfaces/registry.py     Compatibility and configured preference resolution
ai_lab/interfaces/sessions.py     Open, heartbeat, release and expiry
ai_lab/api/routes/direct.py       Read-only choices and explicit session actions
ai_lab/web/js/views/             Display the choices returned by the API
```

`interfaces/` depends on the existing runtime, gateway and host contracts; the
engine package does not import a browser provider. A provider adapter contains
integration details for its UI family, not names of individual checkpoints.
A new model version changes configuration. A new engine or UI family may add
one adapter and its tests.

## Traffic and memory ownership

Two traffic modes have different safety rules:

1. **Gateway client.** The UI calls AI-LAB's stable model API. Each inference
   request takes and releases a gateway lease, exactly as an agent request does.
   Open WebUI talking to the AI-LAB gateway is this case. An idle browser tab
   holds no model memory reservation.
2. **Native runtime UI.** The page calls the engine process directly. Opening
   it creates a time-limited direct session with heartbeat and explicit close.
   The scheduler treats that model or GPU pool as reserved while the session
   can submit work. Expiration releases a lost tab. A native UI must not bypass
   the scheduler while AI-LAB silently evicts its process.

A UI service has its own CPU/RAM lifecycle. Its existence does not imply a
loaded model or a GPU reservation. Several browser tabs may reuse one service.
Several models may stay loaded only when the existing resource planner permits
it; multiple UI sessions do not assert that they fit. A shared ComfyUI process
needs job-level ownership and a known workflow-to-model binding before it can
serve several AI-LAB installations. The current bridge launches one private
ComfyUI process per configured instance, so simply linking its frontend would
bypass AI-LAB's lease and model selection.

## Parameters have four scopes

| Scope | Example | Effect |
|---|---|---|
| Model installation | Checkpoint or voice pack path | Selected from host configuration; new version is a config change |
| Engine launch | Context size, VRAM mode, parallel slots | Saved through AI-LAB, applied by a guarded reload |
| Inference or workflow | Temperature, voice, seed, sampler, prompt | Set per request or job; does not rewrite installation settings |
| UI preference | Theme, panel layout, conversation history | Stored by the UI, never treated as a model parameter |

The provider declares which controls it actually exposes and their scope.
AI-LAB reports values in effect, not merely saved defaults. When the native UI
cannot edit a launch setting, AI-LAB's Settings panel remains the place to do
that. The UI must not imply that a per-request control changed the installed
model.

## Initial provider candidates

These are provider candidates, not claims that their integration is complete.
Each is accepted only after testing on the host that actually has the model.

| Runtime or task | Candidate direct interface | Integration decision |
|---|---|---|
| llama.cpp on Mac and Linux | Built-in `llama-server` UI | Keep it; it belongs to the loaded process. Identify direct-session protection for its non-gateway requests. |
| vLLM on Linux | One Open WebUI deployment connected to AI-LAB's OpenAI-compatible gateway | Share it among vLLM models and, if useful, other text engines; do not launch one UI per model. |
| ComfyUI image, edit, video or music workflows | ComfyUI frontend on the same controlled backend | Expose only after workflow binding, model visibility and scheduler ownership are proven. A shared backend is a later option. |
| Kokoro on Mac | Evaluate Kokoro-FastAPI with its own `/web` and API | Verify Apple GPU operation, checkpoint reuse and AI-LAB lifecycle before replacing the current speech worker. |
| Other speech, music, OCR and analysis engines | Research the engine's maintained UI and API individually | Build an AI-LAB form only if no suitable maintained UI exists or it cannot use the same managed runtime. |

The current `LaunchPlan.web_ui` boolean and the task-wide `TASK_ACTIONS` map
cannot express these choices. Replace the boolean with provider availability
and return the selected provider and an `open` URL from an API endpoint. The
Models page draws only the returned action; it does not choose a UI by task.

## Rollout and acceptance

1. Inventory every configured installation on Mac and Linux. Record its
   inference engine, compatible UI candidates, controls, operating system and
   current agent route. Group the work by provider, but validate each model on
   each installed host.
2. Add provider definitions, host deployments, binding configuration and a
   read-only compatibility API. Existing agent routes and instances remain
   valid. Unknown provider IDs fail validation before a UI link is shown.
3. Add direct sessions and the two traffic modes. Prove that closing a tab,
   losing a heartbeat, a concurrent agent request and a model switch release
   or respect the right reservation.
4. Integrate providers one family at a time. For each installed model, verify
   opening the UI, changing each advertised request parameter, applying a
   launch setting through guarded reload, producing a result, and calling the
   same model by its agent route. The user evaluates output quality manually.
5. Store chosen bindings and deployment settings in private `opts` snapshots;
   compare active Mac/Linux bytes after deployment. Public code contains only
   provider implementations, schemas, examples and tests.

A provider is ready only when its UI opens on the actual host, addresses the
same configured model as the agent route, exposes its advertised controls,
and does not evade AI-LAB's capacity management. A polished UI is not evidence
that those four conditions hold.

## References for provider selection

- [llama.cpp server and built-in UI](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
- [vLLM OpenAI-compatible server](https://docs.vllm.ai/en/latest/serving/online_serving/openai_compatible_server/)
- [Open WebUI's vLLM connection guide](https://docs.openwebui.com/getting-started/quick-start/connect-a-provider/starting-with-vllm/)
- [ComfyUI as interface and inference engine](https://docs.comfy.org/essentials/core-concepts/links)
- [Kokoro model and official demo](https://huggingface.co/hexgrad/Kokoro-82M)
- [Kokoro-FastAPI interface and Apple Silicon instructions](https://github.com/remsky/Kokoro-FastAPI)
