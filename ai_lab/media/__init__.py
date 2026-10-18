"""The manager's job queues for generated media: music, speech and images.

Jobs that can be cancelled and survive a restart (`jobs.py`, `image_jobs.py`,
their store `job_store.py`, and how an image workflow is filled,
`image_workflows.py`). They send their work through the gateway like any other
client. The isolated engine processes that answer them are elsewhere
(`engine_host/` and the engine server scripts).
"""
