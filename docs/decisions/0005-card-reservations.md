# 0005 — Card reservations: a team holds the card for a whole run

Status: accepted, 2026-10-17 (by the owner). Designed with radio-lab/ingest (request `2026-10-17-card-reservations`).

## Context

The gateway hands out the card per request: each call takes a short lease, and a call that names another model makes the gateway swap models (qwen36-nvfp4 reloads in about 3 minutes). A team's pipeline calls the card in several steps with gaps between them; in a gap another team can swap the model, and the pipeline's next step waits for a reload or finds the card taken.

Ingest's hourly collect on CT 109 (measured 2026-10-17, about 4.5 minutes): triage, screen, classify and summarize on qwen36-nvfp4, detect_rewrites and group_stories on bge-m3, with a 2-minute gap (fetch_texts) that can grow past 10 minutes on a big day. Ingest's development copy on the Mac calls the same Linux gateway: debug runs and benchmarks of 5–40 minutes, sometimes with a model swap.

## Decision

1. **A reservation holds the whole card**, not one model. While a reservation is active, only its holder's calls reach this machine's models (the card's and the embeddings models); the holder may load and use any models, in any order. Every other such call waits. Subscription models (Claude Code and the like) run in the cloud and are never held back.
2. **AI-Lab applies its own policy and does not judge its clients.** It does not decide who a client is (Mac or Linux) or what priority it deserves; the client says its priority, 1 (low) to 5 (high), and AI-Lab orders by it. Callers are not checked by address. If clients ever need to be validated, that comes later, for example with a token.
3. **Who holds it.** A reservation names `team` (required), `instance`, `purpose` and `run_id`, all labels for the page and the logs ("ingest @ ct109 · collect · run 20261017-110327").
4. **Order of the queue:** priority, then arrival. **No preemption:** a running reservation is never interrupted; priority only decides who gets the card next.
5. **Aging.** A waiting reservation rises one priority level every `aging_s` (default 10 minutes, 0 = never), up to 5, so nothing waits forever.
6. **Proof of life.** The holder, and every waiting team, sends a heartbeat. No heartbeat for `heartbeat_timeout_s` (default 3 minutes) ends the reservation. AI-Lab cannot call the teams; the heartbeat is the health check.
7. **Unused.** A granted reservation that makes no call within `unused_s` (default 10 minutes) ends. After its first call only the heartbeat counts, so a long gap between steps keeps the card.
8. **Duration.** `max_s` defaults to `default_max_s` (30 minutes). The holder may extend it up to `ceiling_s` (4 hours). Past `max_s` the reservation ends and the card goes to the next in line.
9. **A call under a reservation that is not the holder's** gets 409 with the reason: ended (released, expired, cancelled, unknown) or not granted yet (with its position). It is never served as a call without a reservation.
10. **Calls without a reservation** work as before while the card is not reserved; while it is, they wait up to `unreserved_wait_s` (30 minutes), then get 409 naming the holder. With `required` on, they are refused at once with 428.
10a. **Checked twice: at the door, and again when the call is served.** A call let through the door may wait in the card's own queue behind other calls or a load. When it reaches the head of that queue, before anything is unloaded, the reservations are asked again: a call without a reservation overtaken by a reservation granted meanwhile goes back to the door and waits there (at most 3 times, then 409); a call whose own reservation ended meanwhile gets the 409 it would have got at the door. A call admitted **at once** — its model loaded and a place free — is not checked a second time: that path cannot change the card's models, so it cannot break a holder's run.
11. **Embeddings (bge-m3)** keep their placement rule (ADR 0004): on the card when there is room, on the processor otherwise. While the card is reserved they serve only the holder; who goes next is decided by the reservations' priorities.
12. **Manager restart.** Reservations are saved to disk and restored with what is left of their time; the time the manager was down counts against `max_s` (the limit is a promise to the teams waiting, who could not use the card either). The heartbeat clock starts again at the restart, and the holder sees `restored: true`.
13. **Manual actions.** Load, Unload and Apply on the Models page are refused while the card is reserved, saying by whom, even with `force`; cancel the reservation on the Reservations page first.
14. **Both machines.** The mechanism runs on the Linux and the Mac AI-Lab alike.
15. **Page and settings.** A Reservations tab shows the holder, its time left, the queue with positions and priorities, the recent history, a Cancel button, and every setting above.
16. **Not now:** booked time windows (a scheduler). Can be added later as reservations created at a set time.

## API

- `POST /api/reservations` `{team, instance, purpose, run_id, priority, max_s}` → the reservation, `state` `granted` or `queued`, `position`
- `GET /api/reservations/{id}` → `state` (`queued`, `granted`, `ended`), `reason`, `position`, `time_left_s`, `effective_priority`, `restored`
- `POST /api/reservations/{id}/heartbeat`
- `POST /api/reservations/{id}/extend` `{seconds}`
- `DELETE /api/reservations/{id}` (release)
- `POST /api/reservations/{id}/cancel` (by hand, from the page)
- `GET /api/reservations` → holder, queue, history, settings
- `PATCH /api/reservations/settings`
- Every model call under a reservation carries the header `X-AI-Lab-Reservation: <id>`.

## Consequences

- A pipeline keeps its models between steps; another team cannot swap the model in the middle of a run.
- Priorities are the clients' responsibility; a client that asks for 5 gets 5.
- A team that forgets its heartbeat loses the card after 3 minutes.
