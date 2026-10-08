# 0002 — opencode as the gate to cloud open-source models

Status: accepted, 2026-10-08. Extends 0001.

## Context

The owner wants open-source models run in the cloud reachable through the
gateway like every other model. opencode already knows how to reach many
such providers, and its own configuration says which providers and models
exist. Configuring them there is the comfortable way for the owner.

## Decision

1. opencode is a third dialect of the subscription lane (`OpencodeCli`):
   `opencode run -m <provider/model>`, prompt on standard input, the answer on
   standard output.
2. The gateway does not repeat opencode's provider configuration. An entry in
   `providers.models` only names one model opencode already knows:
   `{"vendor": "opencode", "model": "<provider/model>"}`.
3. The same rules as the other vendors apply: one gate for the vendor, the
   same concurrency, launch spacing, rate-limit pauses, retries and timeout.
4. Every tool is denied for the call (`OPENCODE_CONFIG_CONTENT` with
   `{"permission": {"*": "deny"}}`), so the agent can only answer. Checked on
   CT 102 with opencode 1.18.35: a prompt asking it to run a shell command
   created no file.
5. Like Claude Code and Codex, opencode runs in the container on its CPU. It
   never uses the graphics card, so it neither waits for nor disturbs the
   local models.

## Consequences

- An unknown model name comes back from opencode as an unspecific "Unexpected
  server error": it is treated as transient and retried before failing.
- Signing in to the cloud providers is done in opencode, in the vendor's home,
  by the owner.
