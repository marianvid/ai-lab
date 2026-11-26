# 0001 — Subscription CLIs as gateway providers

Status: accepted, 2026-10-07.

## Context

Some work needs a frontier model: ingest's audit of how it classifies its
sources gets too much wrong on the local models. The owner has two
subscriptions that include such models, reachable only through their command
line programs: Claude Code (`claude -p`) and Codex (`codex exec`). Every
project already reaches its models through the AI-Lab gateway, in the OpenAI
shape, and the console's LLM Map chooses a model per slot by gateway and name.

Both subscriptions are limited by the vendor, not per model: Sonnet and Opus
spend from one Anthropic allowance, every Codex model from one OpenAI
allowance. When one caller reaches the limit, every other caller of the same
vendor will be refused too.

## Decision

1. The gateway serves the subscription models under names of their own
   (`claude/sonnet`, `codex/gpt-5.6-terra`, …), on the same
   `POST /v1/chat/completions` as every local model. Callers change nothing
   but the name.
2. They run in a lane of their own, `ai_lab/providers/`, beside the card's
   scheduler rather than inside it. A CLI call uses no card: it must neither
   wait behind a local model being loaded nor hold one up.
3. One gate per **vendor**, not per model (`VendorGate`): a limit on calls in
   flight, a minimum interval between two launches, and a shared pause after
   a rate limit, so the first caller to meet the limit stops the others from
   spending calls on it.
4. Errors are classified from the CLI's own text (`errors.classify`), because
   a CLI reports everything as an exit code plus prose: rate limit (wait,
   retry the same model), fatal (answer at once, so the caller's fallback
   takes over), transient (retry a few times). Unrecognised text is transient
   and counted under its own name, so a vendor rewording its messages shows
   as a growing counter rather than as silence.
5. Each CLI is a Strategy behind one interface (`dialects.ClaudeCli`,
   `dialects.CodexCli`). The prompt goes in on standard input, never as an
   argument: a long prompt would pass the operating system's limit on one
   argument.
6. Limits, binaries, homes and the model list are configuration
   (`providers` in `config.json`). The defaults in code are the values
   measured in production on the owner's earlier project: 6 calls at once per
   vendor, 0.5 s between launches, 30 s → 120 s → 300 s after a rate limit
   with at most 3 retries, network failures retried after 5 s and 15 s, 600 s
   per call.
7. Requests and failures are counted per vendor per day in the state
   directory and shown in `GET /api/gateway`, so the shared allowance is
   watched in one place.
8. `reasoning_effort` is passed to the CLI. A request that names none gets
   the vendor's `default_effort` (`medium` for Claude Code on Linux). An
   effort a CLI lists as unknown is refused with 400 rather than sent on:
   Claude Code ignores one with only a warning and runs at another effort.
9. When `response_format` asks for JSON and a model still answers with one
   Markdown code block (```` ```json … ``` ````), the gateway returns only the
   inside. Measured on 2026-10-11: a small model did it in 3 of 49 answers,
   and a client expecting plain JSON could not read them. Any other answer is
   passed on unchanged.
10. A client's cancel is the server's cancel (owner, 2026-11-26: "Daca
    clientul (news) face cancel atunci si serverul (ai-lab) face cancel daca
    acest lucru mai este posibil"). The cancel travels as the client closing
    its connection, as the CT 109 scheduler does for the running calls of a
    cancelled lot. While a call waits (for a place, a vendor's pause, a retry)
    or its CLI runs, the gateway looks every `client_check_s` (1 s) at the
    client's socket without reading it. Client gone: a waiting call leaves
    without starting; a running CLI's whole process group gets `SIGTERM`,
    then `SIGKILL` after `stop_grace_s` (5 s); the place is freed at once.
    The call is counted `cancelled` beside the failures, and the vendor's
    pause never hears of it. What the vendor had already used is spent.
    Before this, a started call ran on up to `timeout_s` (600 s) for nobody.
11. A vendor at its limit is said, and a call may refuse to wait for it
    (owner, 2026-11-26: "banda din consola - da, construieste-o"; "construieste
    acel nu ma tine in loc").
    - `GET /api/gateway` gives `providers.<vendor>.limit`: `state`
      (`ok` / `limited` / `unknown`), `kind` (`usage`: the subscription's
      allowance; `rate`: a short refusal), `since`, `retry_at` (AI-Lab's own
      next try, the end of the vendor's pause), `resets_at` (the vendor's own
      reset time, only when its words carry one AI-Lab can read: an ISO time or
      Claude Code's `|<epoch>`), `model`, `said`. Learned from real calls only,
      kept in the state directory beside the logins. A usage refusal is
      `limited` at once; a rate refusal from the `limited_after_strikes`-th in
      a row (2), so one short pause does not show.
    - A call with `"ai_lab": {"hold": false}` in its body is never held through
      a vendor's pause: refused at once with 503 `vendor_limited` (`ai_lab.detail`:
      `vendor`, `machine`, `kind`, `retry_at`, `resets_at`), when it arrives
      during a pause, when a pause begins while it waits for a place, or when
      the vendor refuses it for a limit (no retry). Waiting for a free place
      alone is not a refusal. Without the field nothing changes.

## Consequences

- No streaming: a CLI answers when it is done. A request asking for a stream
  is refused by name.
- No images and no tool calls through this lane; text in, text out.
- Signing in is done once per machine, by hand, in the configured homes.
- Usage figures are requests, not tokens: the CLIs do not report tokens in
  the form used here.
