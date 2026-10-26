# Who may call AI-Lab: tokens, the heartbeat, login

The approved list of 2026-10-25 (`claude/requests/2026-10-25-simplify-ai-lab/list.md`, A1, A2, A7, A8) in plain words.

## Tokens (A2)

- Every client the owner registers gets **one token**, shown once, when it is registered (`POST /api/clients {"name": "scheduler-linux"}`). AI-Lab keeps only its SHA-256 digest, in the file clients.json in the host's state folder (mode 0600).
- A client sends it on every call: `Authorization: Bearer <token>`.
- Open without anything: `GET /api/health` ("is it up", and whether AI-Lab is On), plus what the login page needs (`GET /api/session`, `POST /api/login`).
- **Settings → Security, `require_token`:**
  - off (the switch-over, step 1): a call without a valid token is still served; the door writes one line in the journal per caller kind and route;
  - on (step 4): it is refused with `401`.
- The web interface is let in by its session cookie, not a token.

## Refusals (A7)

OpenAI-shaped on `/v1/...`, a sentence elsewhere; the code is in both:

    {"error": {"message": "...", "type": "ai_lab", "code": "<code>"}, "ai_lab": {"reason": "<code>", ...}}

| HTTP | code | when |
|---|---|---|
| 401 | `token_missing` | no token, and tokens are required |
| 401 | `token_invalid` | a token no registered client has |
| 503 | `ai_lab_off` | AI-Lab is switched Off |
| 409 | `model_replaced` | the call's model was replaced for another call |

## The heartbeat (A8)

`POST /api/clients/heartbeat` with the scheduler's token, once a minute:

    {"map": {"made_at": "...", "cards": [{"card": "...", "reservations": [...], "running": [...], "queued": [...]}]}}

Answer: `{"received_at": <unix time>, "on": true|false}`. AI-Lab keeps the last map per client in memory only and shows it; it enforces nothing. A client that has sent a heartbeat shows as down after 3 minutes of silence; one that never sends any (voice-studio, the owner's tests) is never "down". `GET /api/clients` lists every client with `last_call_at`, `heartbeat_at`, `down` and `map`.

## Login (A1)

- **Settings → Security**: set the password (`POST /api/security/password {"new", "current"}`), then switch `login_required` on (`PATCH /api/security`). Login cannot be required before a password exists.
- The password is kept as a salted scrypt hash in the file security.json, never as text.
- `POST /api/login {"password"}` sets the session cookie (`HttpOnly`, `SameSite=Strict`, 30 days); `POST /api/logout` clears it. Sessions live in memory: a restart asks for the password again.
- **"No logon"** (`login_required` off): opening the page starts its session by itself.

## On/Off (A6)

`POST /api/power {"on": false}` switches AI-Lab Off (the Access page has the button):

- every call that would use a model (POST `/v1/...`, `/tokenize`, Load) is refused with `503 ai_lab_off` and `since`;
- whoever waits in the card's queue is turned away;
- the calls running end within about a second;
- reading stays open, the model list included. Off survives a restart; `{"on": true}` switches it back.

## The operation log (A9)

What AI-Lab is doing, one line per step: a model starting or stopping, a model put on the card and what was taken off for it, the queue turned away, a call without a valid token, On/Off. The page reads the newest 500 lines (`GET /api/journal?after=<id>`); every line is also written to logs/operations.log in the host's state folder, rotated at 5 MB, five files kept.

## Models and the card (A3, A5)

- Calls are served first come, first served; a loaded model takes as many at once as it has places. Who may use the card when is the schedulers' business: AI-Lab keeps no reservations.
- A model that does not fit beside the others replaces one: the first, idle before busy and used longest ago first, whose place alone is enough; only when none is enough do several go. A model still answering goes only after its calls end: replacement never cuts a call.

## The Mac's memory ceiling (A12)

`memory.ceiling_mb` (the Machine card, "Ceiling for models"; 0 = none), on unified memory only: a model may take the smaller of what is really free when it loads and what the ceiling leaves above what the models already hold.

## Images and music (A10)

Ordinary calls: `POST /v1/images/generations`, `/v1/images/edits` (by named profile, `GET /api/image-profiles`), `POST /v1/audio/music/generations`. No job list, job id or polling. Each kind has its own first-byte limit in `gateway.task_timeouts`, counted from the moment the call reaches the engine.

## Engine ports (E)

Every engine's API (vLLM, llama-server, the image, speech, music, audio and
embeddings servers) listens on `127.0.0.1` only (`network.LOOPBACK`). Another
machine reaches a model only through AI-Lab's own port, with a token.
AI-Lab, the benchmark and the health checks call the engines on the same
machine, so they are not affected.

The pages a person opens from an engine's button (Music, Speak, Create, Edit:
ComfyUI, the studios, the editors) listen on `network.PAGE_INTERFACES`, today
every network card. llama-server's chat page shares its API port, so its Chat
button works only in a browser on the same machine.

