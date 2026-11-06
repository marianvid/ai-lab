# Music generation

AI-Lab has several music engines. Each one is a separate program that turns a
text description (and usually lyrics) into a song. They all answer the same
address, so a caller does not need to know which engine is behind a model name.

| Engine | What it does | Linux | macOS | Direct interface |
|---|---|---|---|---|
| ACE-Step 1.5 | Song or instrumental from a style prompt and optional lyrics | yes | yes | ACE-Step's own Gradio page |
| HeartMuLa | Song from style tags and lyrics | yes | no | HeartMuse Studio |
| YuE2 | Song from a style and lyrics; also writes an ABC score | yes | yes | ds-yue-webui Studio |
| MuLaCover | New vocal version (a "cover") of an uploaded WAV | yes | no | none |
| Khala | Song or instrumental, length chosen as a "bucket" | no | yes | Khala Studio |
| ComfyUI Music | Runs a ComfyUI workflow, such as MiniMax Music 3 | yes | no | Native ComfyUI |
| LeVo 2 | Song, or an instrumental when no lyrics are sent. **Research licence only: never for broadcast or any commercial use** | yes | yes¹ | none |
| Stable Audio 3 | Instrumental music or sound from a text prompt, up to 380 s | yes | yes | none |

¹ On macOS, LeVo2.cpp is built with Metal. Metal has no `DIAG_MASK_INF`, so the
build used there makes the same causal mask from `FILL` and `TRI` and passes it
to `soft_max_ext`. The patch is in the machine's own copy of LeVo2.cpp, not in
this repository.

"Linux" and "macOS" say which host the engine is offered on
(`hosts/linux.py` and `hosts/darwin.py`). An engine still needs its runtime
installed and configured there. Machine paths and model selection belong to
private `opts`; the public repository supplies only the adapters.

Every engine accepts only models listed in its own configuration. The manager
cannot load an arbitrary safetensors directory as a music model. This prevents
a Library label from presenting an unusable Generate button. For ACE-Step the
model name must map to a checkpoint in ACE-Step's own `checkpoints` folder.

To use a model by hand, load it on **Models** and press **Music**. That opens
the engine's own full interface, not a reduced AI-Lab form. See
[Use a loaded model directly](direct-use.md).

Agents use the same gateway:

```http
POST /v1/audio/music/generations
Content-Type: application/json

{"model":"music-ace-xl","prompt":"Ambient piano, no vocals",
 "duration":15,"instrumental":true}
```

The response contains `data[0].b64_wav` — the WAV file written as base64 text
— and `mime_type: audio/wav`. It also carries the `seed` that was used (the
number that fixes the random choices; the same seed and input give the same
song) and the real `duration` in seconds. The call is synchronous: the
connection stays open until the song is finished. Except for YuE2, which hands
the work to its Studio, the adapter deletes its temporary working files
afterwards. Save the result if it is needed later.

The call is an ordinary call: it waits in the card's queue like any other
and is answered when the song is made (there are no background jobs since
2026-10-25). Its first-byte limit is `gateway.task_timeouts.music-generation`.

## What each engine accepts

Every engine takes `model`, `prompt` (the style, 1–4000 characters) and an
optional `seed` (0 to 4294967295). A field an engine does not know is refused
with an error, not silently dropped.

| Engine | Other fields | Rules |
|---|---|---|
| ACE-Step | `lyrics`, `duration`, `instrumental`, `vocal_language` | duration 5–180 s, default 30 |
| HeartMuLa | `lyrics`, `duration`, `instrumental` | lyrics required; `instrumental` must be `false`; duration 5–180 s, default 60 |
| YuE2 | `lyrics`, `instrumental`, `abc` | lyrics required; `instrumental` must be `false`; no duration |
| MuLaCover | `lyrics`, `instrumental`, `reference_audio_base64` | lyrics and a WAV of at most 25 MiB required; `instrumental` must be `false` |
| Khala | `lyrics`, `instrumental`, `vocal_language`, `length_bucket` | language one of Chinese, English, Japanese, Korean, Cantonese |
| ComfyUI Music | `lyrics`, `instrumental`, `duration` | `instrumental` (true or false) must be sent; lyrics required unless instrumental; duration 5–120 s, default 30 |
| LeVo 2 | `lyrics`, `instrumental`, `duration` | lyrics required unless `instrumental: true`; duration 5–300 s, default 60 |
| Stable Audio 3 | `instrumental`, `duration`, `negative_prompt` | instrumentals only (`instrumental` may be omitted or `true`, no `lyrics`); duration 1–380 s, default 30; put the tempo (BPM) in the prompt |

A few engine-specific details:

- **YuE2** can take an ABC score in `abc`. ABC is a plain-text way of writing
  down music notes. The answer returns the score it used in `score_abc`, and
  a `truncated` flag. YuE2 reports truncation per stage; `truncated` is true
  only when some stage actually cut the song short. It also returns
  `generation_id`: the name of the song's folder in YuE2 Studio, which a
  window edit names as its `source` (below).
- **Khala** measures length in buckets: numbered length steps instead of
  seconds. The allowed range comes from the model's configuration. The answer
  repeats the `length_bucket` used.
- **MuLaCover** needs the source song as a WAV file in base64 in
  `reference_audio_base64`.

`GET /api/instances` shows, under `music_form`, what each loaded music model
needs — for example whether lyrics are required and which kind of duration it
takes.

## YuE2: regenerating one window of a song

`POST /v1/audio/music/window-generations` takes a YuE2 song made earlier and
regenerates only a window of it (for example a line whose words were sung
wrongly), once per candidate seed. Each candidate is the whole song again, with
only that window new, kept as a song of its own with its own `generation_id`.
The one kept can be the `source` of the next edit. The recipe is the one
radio-lab/music proved in its lab on 2026-11-05: the new tokens are written for
the window only, and the audio around it is held to the original (RePaint), so
outside the window the song does not change. The work runs in YuE2 Studio's
`window` job, on the model already loaded (the Studio patch:
`system/engines/ds-yue-webui-window.patch`).

```
{
  "model": "<the YuE2 entry>",
  "request_id": "<chosen by the caller, 1-64 of A-Z a-z 0-9 . _ ->",
  "source": "<generation_id of the song to edit>",
  "window": {"start_s": 41.0, "end_s": 56.4},
  "lyrics": "<the whole lyrics, the same section tags, the window's lines edited>",
  "candidates": [{"seed": 101}, {"seed": 202}],
  "context_s": 5.0, "edge_s": 0.4, "crossfade_s": 0.3,
  "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 100,
               "repetition_penalty": 1.2, "penalty_window": 50}
}
```

- `context_s`, `edge_s`, `crossfade_s` and `sampling` (or any part of it) may be
  left out: the instance's settings apply. The window's seconds are rounded to
  YuE2's frames (1/25 s). Style and score stay the source's: a changed score
  is a new full song, not a window.
- It is an ordinary call, like the song call: it waits for the card and is
  answered when every candidate is done, within the same first-byte limit
  (`gateway.task_timeouts.music-generation`). About two minutes per candidate
  for a 30 s window on the Mac (an estimate until it is measured).
- The answer lists the candidates, without audio: `request_id`, `source`,
  `window_s`, `clip_s` (the window plus its context), `noise_seed` (the first
  song's seed, kept through every edit), `cancelled`, and per candidate `seed`,
  `generation_id`, `tokens_changed_pct`, `short_window` (true when fewer new
  tokens came than the window holds and the rest was taken from the original),
  `seconds` and `timings`.
- Refused with **422** and the reason: no usable `request_id`; an unknown
  field; a window outside the song or outside the allowed length; a source that
  is unknown or whose files are gone; lyrics whose section tags differ from the
  source's; more candidates than allowed. A `request_id` already waiting or
  running: **409**.
- `DELETE /v1/audio/music/window-generations/{request_id}` stops a call. Still
  waiting for the card, it is dropped from the queue (`"was": "waiting"`);
  running, no further candidate starts and the one running is cancelled
  (`"was": "running"`); the candidates already done are in its answer. A
  cancel that lands as the call is handed to the engine still holds: the call
  then answers `cancelled` with no candidates and sends nothing. A client that
  only closes its connection does not stop the work: call this.

The songs' files, read from the engine's folder (the model need not be loaded).
These calls have no body, so they name the YuE2 entry with `?model=`:

| Call | What |
|---|---|
| `GET /v1/audio/music/generations/{generation_id}/audio?model=<entry>` | the whole song, FLAC |
| `GET /v1/audio/music/generations/{generation_id}/window-audio?model=<entry>` | the regenerated clip (window plus context), WAV; a window candidate only |
| `DELETE /v1/audio/music/generations/{generation_id}?model=<entry>` | removes the song, for a candidate not kept; nothing expires on its own. Through YuE2 Studio while YuE2 is loaded (it refuses a song a job is still writing: 409), from the folder when it is not. The source of a window call still waiting or running is refused: 409 |

The defaults and limits are the YuE2 instance's settings, shown with their
explanations on its page (`window_*`): context 5.0 s, edge ramp 0.4 s,
crossfade 0.3 s, swap margin 2.5 s (how far outside the window the new audio
replaces the old; within the context), at most 5 candidates, a window of
1-60 s, and the window's sampling (1.0 / 0.95 / 100 / 1.2 / 50).

---

[← all documents](../README.md)  ·  [Use a model directly](direct-use.md)
