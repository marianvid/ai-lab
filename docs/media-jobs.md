# Media jobs

Music, speech and video models can be called synchronously by an agent
through their `/v1/...` routes: `/v1/audio/music/generations`,
`/v1/audio/speech/generations` and `/v1/videos/generations`. The connection
then stays open until the file is made. A media job is the other way: the
request returns at once with a job ID, the work runs in the background, and the
result is kept on disk to be fetched later — also after the caller has gone
away.

The browser page no longer has a form that uses these jobs. The AI-Lab
Workbench that offered **Generate in background** was removed; a loaded model's
**Music**, **Speak** or **Video** button now opens the engine's own interface,
where the engine has one, instead (see
[Using a model directly](direct-use.md)). Jobs are an API for agents and
scripts.

`POST /api/media-jobs` accepts exactly three fields — a configured model ID,
its task, and the same engine input body used by the direct route:

```json
{"model":"my-music-model","task":"music-generation",
 "input":{"prompt":"quiet piano","duration":20}}
```

Tasks are `music-generation`, `speech-synthesis`, and `video-generation`.
A model name nobody serves is refused at once, not turned into a job that fails
later. The response is the job itself: `id`, `model`, `task`, `status` and
timestamps. `GET /api/media-jobs` lists metadata without
large result bytes; `GET /api/media-jobs/{id}` includes the result when the job
succeeds — the same JSON the direct route would have returned.
`DELETE /api/media-jobs/{id}` cancels a queued job or requests
cancellation of a running job.

A job's `status` is one of `queued`, `running`, `cancelling` (asked to stop,
still running), `succeeded`, `failed` or `cancelled`. A failed job says why in
`error`.

Jobs run **one at a time**, oldest first. Each one takes its place in the
[Gateway](gateway.md) queue like any other request, so the model is loaded if
needed and nothing unloads it mid-generation. The wait for the engine's answer
uses the first-byte limit set for that task (see
[Gateway behavior](gateway-behavior.md#how-long-to-wait-for-an-engine)).

Jobs and outputs live under the host's private state directory in `media-jobs/`.
Metadata is written atomically, and files are accessible only to the manager
account. A manager restart marks unfinished jobs failed; completed outputs
remain available until `media.result_ttl_s` expires. Cleanup runs at startup
and after each job. Source prompts and uploaded media are held only while a job
runs, never in the saved metadata.

The limits live in the `media` section of the configuration. Any other name
there is refused when the configuration is loaded.

| setting | what it limits | default | allowed |
|---|---|---|---|
| `result_ttl_s` | how long a finished job and its result are kept | 86,400 s (one day) | 60 s to 30 days |
| `max_queue` | jobs not yet finished; one more is refused | 32 | 1 to 1,000 |
| `max_input_bytes` | size of the `input` object, as JSON | 40 MiB | 1 KiB to 512 MiB |
| `max_result_bytes` | size of the engine's answer; larger fails the job | 256 MiB | 1 KiB to 1 GiB |

ComfyUI workers support a running-job interrupt. Other engines may finish their
current generation after cancellation; the gateway lease stays held until they
finish, so another model is not loaded over an active generation.
