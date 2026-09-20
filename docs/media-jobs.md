# Media jobs

Music, speech and video models can still be called synchronously through their
`/v1/...` routes by an agent. The Workbench also offers **Generate in
background**. It submits a durable job, shows its state, and lets a person
reopen the saved result after a browser reload.

`POST /api/media-jobs` accepts a configured model ID, its task, and the same
engine input body used by the direct route:

```json
{"model":"my-music-model","task":"music-generation",
 "input":{"prompt":"quiet piano","duration":20}}
```

Tasks are `music-generation`, `speech-synthesis`, and `video-generation`.
The response contains a job ID. `GET /api/media-jobs` lists metadata without
large result bytes; `GET /api/media-jobs/{id}` includes the result when the job
succeeds. `DELETE /api/media-jobs/{id}` cancels a queued job or requests
cancellation of a running job.

Jobs and outputs live under the host's private state directory in `media-jobs/`.
Metadata is written atomically, and files are accessible only to the manager
account. A manager restart marks unfinished jobs failed; completed outputs
remain available until `media.result_ttl_s` expires. Cleanup runs at startup
and after each job. The private configuration also sets queue, input and result
size limits. Source prompts and uploaded media are held only while a job runs,
never in the saved metadata.

ComfyUI workers support a running-job interrupt. Other engines may finish their
current generation after cancellation; the gateway lease stays held until they
finish, so another model is not loaded over an active generation.
