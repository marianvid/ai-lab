# Reserving the card for a run

When a pipeline calls this machine's models in several steps, reserve the card
for the whole run. While you hold it, nobody else's calls reach the models
here, so nobody can swap the model you are using between your steps. The rule
and its reasons: [ADR 0005](decisions/0005-card-reservations.md).

## The five steps

1. **Ask:**

   ```
   POST /api/reservations
   {"team": "ingest", "instance": "ct109", "purpose": "collect",
    "run_id": "20261017-110327", "priority": 4, "max_s": 1800}
   ```

   The answer has an `id` and a `state`: `granted` (the card is yours) or
   `queued` (with your `position`). Priority goes from 1 (low) to 5 (high).

2. **Wait, if queued:** ask `GET /api/reservations/{id}` every 15 s or so until
   `state` is `granted`. Keep sending heartbeats while you wait.

3. **Call the models** as usual, adding the header
   `X-AI-Lab-Reservation: <id>` to every call.

4. **Heartbeat** every 30 s while the run lives:
   `POST /api/reservations/{id}/heartbeat`. Three minutes without one ends the
   reservation. Need more time than you asked for?
   `POST /api/reservations/{id}/extend {"seconds": 900}` (up to the ceiling).

5. **Release** at the end, also after a failure or a stop:
   `DELETE /api/reservations/{id}`.

## What can go wrong

| Answer | Meaning | What to do |
|---|---|---|
| 409, "has ended: …" | Your reservation ended (no heartbeat, unused, past its time, cancelled from the page). | Ask for a new one; log the reason. |
| 409, "not granted yet (position N)" | You called before it was granted. | Wait for `granted`. |
| 409, "the card is reserved by …" | A call without a reservation waited too long. | Reserve, or try later. |
| 428 | This AI-Lab requires reservations. | Reserve first. |

`GET /api/reservations/{id}` shows `restored: true` when the manager restarted
during your reservation; the reservation carries on with its remaining time.

The settings (heartbeat limit, durations, aging, whether reserving is
required) are on the Reservations page.
