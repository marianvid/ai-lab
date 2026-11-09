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
| YuE2 | `lyrics`, `instrumental`, `abc`, `cot`, `cfg_scale` | lyrics required; `instrumental` must be `false`; no duration; `cot` (the plan YuE2 writes before the audio) `full` = melody and chords or `melody` = melody only, default the instance's `cot` option; `cfg_scale` (how strictly the style is followed) 0–20, 1.0 = no guidance, default the instance's `cfg_scale` setting (1.0); other values are refused, never changed. A window call takes neither: it keeps the source song's own |
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
| `DELETE /v1/audio/music/generations/{generation_id}?model=<entry>` | removes the song, for a candidate not kept; nothing expires on its own. Through YuE2 Studio while YuE2 is loaded (its refusal of a song a job is still writing, which the Studio sends as a 404 with that reason, is answered 409), from the folder when it is not. The source of a window call still waiting or running is refused: 409 |

The defaults and limits are the YuE2 instance's settings, shown with their
explanations on its page (`window_*`): context 5.0 s, edge ramp 0.4 s,
crossfade 0.3 s, swap margin 2.5 s (how far outside the window the new audio
replaces the old; within the context), at most 5 candidates, a window of
1-60 s, and the window's sampling (1.0 / 0.95 / 100 / 1.2 / 50).

## YuE2: moving a song to another AI-Lab

A YuE2 song (a generation) can move between machines, for example from Linux
to the Mac where the music page edits it. The machine that should have it
pulls a copy and checks it. AI-Lab never deletes the other copy: the caller
(music) does that itself once the pull answers `checked: true`.

| Call | What |
|---|---|
| `GET /v1/audio/music/generations/{generation_id}/bundle?model=<entry>` | the song's folder as an uncompressed tar. Its first member is `MANIFEST.json`, `{"generation_id", "files": [{"name", "size", "sha256"}]}`; the files follow, flat, in that order. 404 unknown; 409 while it is the source of a window call or a YuE2 Studio job works on it |
| `POST /v1/audio/music/generations/{generation_id}/pull?model=<entry>`, body `{"from": "<peer>"}`, header `X-Run-Id` | fetches that bundle from the peer through this machine's scheduler door, writes it into a hidden folder beside the songs, checks every size and sha256 against the manifest, and renames it into place in one step |

The AI-Labs never talk to each other directly. The fetch goes to this
machine's scheduler, at
`{scheduler}/ai/<peer>/v1/audio/music/generations/{id}/bundle?model=<entry>`.
The scheduler passes it to the partner scheduler, which passes it to its own
AI-Lab. It carries the caller's `X-Run-Id` (the run the door checks) and no
token: the door asks none of its callers.

The pull answers `{"generation_id", "machine", "files", "bytes", "checked":
true}`. Refusals:

- **400**: no `X-Run-Id`, a peer that is not configured, or a field other
  than `from` in the body.
- **404**: the peer does not have the song.
- **409**: the song is already on this machine, is being edited here, or the
  peer refused it as in use.
- **502**:
  - a copy that fails its check (it is removed; the source is untouched);
  - the door's own refusal (an answer with a `scheduler` key, such as
    `run_stopped`, `unknown_provider` or `partner_lost`), named in the
    message; a door's 404 is never taken for "the peer has no such song";
  - a door that does not answer.

Both calls work whether YuE2 is loaded or not.

The scheduler and the other machines are configured in the YuE2 engine's
section, as `transfer` (`engines.yue2.transfer`, see
`ai_lab/engines/music/transfer.py`):

| Setting | Default | What it is |
|---|---|---|
| `machine` | empty | the name this AI-Lab goes by, given back as `machine` |
| `scheduler` | empty | this machine's scheduler (`http(s)://host:port`), whose model door reaches the other machines. Needed as soon as `peers` names one |
| `peers` | none | the other machines' providers a song may be pulled from: their names in the LLM Map, as the door knows them (e.g. `["ai-lab-linux"]`). Empty: every pull is refused (400, "no peer") |
| `timeout_s` | 60 | how long the door may stay silent while a song is fetched, in seconds. Through the partner, the answer starts only once the partner holds the whole song (~50 MB), so this covers the whole fetch on the other side |
| `chunk_bytes` | 1048576 | the piece a song is read, written and checked in, in bytes (4096 to 64 MiB) |
| `list_most` | 200 | the most generations one list call answers (1 to 10000). 200 covers the Mac's Studio today (169 folders) in one call |
| `put_back_most_bytes` | 1073741824 | the largest bundle a put-back takes, in bytes (1 MiB to 8 GiB), judged by the request's `Content-Length` before anything is read. A song's bundle is about 25–60 MB, so 1 GiB is ample |

## YuE2: the Studio's generations, and putting one back

YuE2 Studio is only a working place: a song stays there while music may still
regenerate its windows. When a song leaves it, music keeps the song's bundle
(the `bundle` call above) in the song's own folder and deletes it from the
Studio. These two calls show what is in the Studio, and bring a kept song
back when it is to be edited again.

| Call | What |
|---|---|
| `GET /v1/audio/music/generations?model=<entry>[&limit=<n>][&offset=<k>]` | this machine's generations, newest first. A read: nothing is changed. `limit` 1 to `list_most` (default `list_most`), `offset` from 0 (default 0); any other value is refused (400) |
| `POST /v1/audio/music/generations/{generation_id}/put-back?model=<entry>`, body the bundle with `Content-Type: application/x-tar`, header `X-Run-Id` | puts a kept bundle back under the same id: every file is checked against the manifest exactly as a pull checks it, written into a hidden folder, and renamed into place in one step |

The list answers `{"machine", "total", "offset", "limit", "generations": [...]}`.
Each generation has:

- `generation_id`, the folder's name;
- `made_at`, when it was finished (the time of its `result.json`; of the folder
  while it has none), ISO 8601 in UTC;
- `bytes` and `files`, its size on disk;
- `complete`, whether the Studio finished it;
- `kind`, `song` or `window` (a window regenerated from another song), with
  `source`, that song's id, for a window;
- `seconds`, the audio's length;
- `style` and `lyrics_start`, the first 300 characters of the request's style
  and lyrics, to recognise it by;
- `seed`, `cot` and `cfg_scale`, as the request gave them (`null` when it did not).

The put-back answers `{"generation_id", "machine", "files", "bytes", "checked":
true}`. Refusals:

- **400**: no `X-Run-Id`; a body that is not `application/x-tar`; a body of 0
  bytes or larger than `put_back_most_bytes`; a bundle that fails its check (it
  does not start with its manifest, its manifest names another generation, a
  file is missing, extra or different, or it is not a tar). Nothing of a
  refused bundle is kept.
- **409**: a generation with that id is already on this machine, or it is the
  source of a window call or a YuE2 Studio job works on it. An existing song is
  never overwritten.

Like the other calls, a caller on the other machine reaches it through its own
scheduler door (`{scheduler}/ai/<provider>/v1/audio/music/...`). A put-back to
the other machine carries the whole bundle (about 25–60 MB) through the
partner exchange between the schedulers, which today gives each exchange
10 seconds: enough on the home network's cable, tight on a weak Wi-Fi.

Both calls work whether YuE2 is loaded or not.

---

[← all documents](../README.md)  ·  [Use a model directly](direct-use.md)
