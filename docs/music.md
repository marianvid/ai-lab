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

For a job that survives a closed connection, and can be listed and cancelled,
use [media jobs](media-jobs.md) with task `music-generation`.

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

A few engine-specific details:

- **YuE2** can take an ABC score in `abc`. ABC is a plain-text way of writing
  down music notes. The answer returns the score it used in `score_abc`, and
  a `truncated` flag. YuE2 reports truncation per stage; `truncated` is true
  only when some stage actually cut the song short.
- **Khala** measures length in buckets: numbered length steps instead of
  seconds. The allowed range comes from the model's configuration. The answer
  repeats the `length_bucket` used.
- **MuLaCover** needs the source song as a WAV file in base64 in
  `reference_audio_base64`.

`GET /api/instances` shows, under `music_form`, what each loaded music model
needs — for example whether lyrics are required and which kind of duration it
takes.

---

[← all documents](../README.md)  ·  [Media jobs](media-jobs.md)  ·  [Use a model directly](direct-use.md)
