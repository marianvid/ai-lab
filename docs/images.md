# Images

Open **Images** in AI-Lab to run a configured ComfyUI workflow without an
agent. Choose a named profile, write a prompt and submit it. Edit profiles also
ask for a source image. The job list shows progress; **View** opens a finished
result and offers a download.

Profiles are defined under `images.profiles` in the active configuration. Each
one names an AI-Lab model entry and an API-format ComfyUI workflow. The public
repository contains example workflows; the actual Linux and macOS profiles
and paths are kept in the private `opts` repository. A model's presence in
Library alone does not make it runnable: it also needs a configured entry,
engine and workflow profile.

The **Models** page still controls loading and settings. The Images page uses
the gateway to load its model when a job starts. ComfyUI's own interface is
not exposed by this adapter; AI-Lab runs the named workflow and returns the
result.

Music models currently have catalog labels only. There is no music engine,
job API or browser studio yet, so they cannot be tried through AI-Lab.

[← all documents](../README.md)
