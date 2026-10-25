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
