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

## Consequences

- No streaming: a CLI answers when it is done. A request asking for a stream
  is refused by name.
- No images and no tool calls through this lane; text in, text out.
- Signing in is done once per machine, by hand, in the configured homes.
- Usage figures are requests, not tokens: the CLIs do not report tokens in
  the form used here.
