# Settings — the machine, its engines, and where models live

Nothing on this page moves on its own. What is used, how warm the card is, how
many requests are running are facts about right now, and right now is the
Gateway page — a figure read here is still true an hour later.

![Settings](screenshots/settings.png)

**Machine** is what this machine is, and the one thing that decides how many
models fit: how much of its memory is held back for the machine itself. A
dedicated card is used whole — nothing else here wants it. The machine's own
memory is shared with the browser and the operating system, so a reserve is
kept out of it. Set as a reserve rather than as an allowance, because a reserve
stays right when the machine is given more memory.

**Engines** shows what is installed, its version, and the state of its update
source. `No update available` means upstream was read and matches what is
installed; `Update status unavailable` means it could not yet be read;
`Updates not managed` means AI-Lab does not install that engine at all. When a
new version exists it is named beside the **Update…** button. No button here
updates anything directly: it opens what the update would bring, and the real
Update is at the foot of that. An engine that arrives as packages but is not
installed yet shows **Install…** instead, which works the same way. See
[Updating an engine](engines.md).

An engine that cannot run carries a short reason, such as `Requires an NVIDIA
GPU` or `Not installed`. An engine this kind of machine does not support at
all — vLLM on a Mac, say — is not listed. While an engine is being built its
output streams into a pane under it; afterwards that pane folds away as
**Last update log**. ComfyUI's add-ons (custom nodes, extra pieces ComfyUI
loads) are listed in a fold of their own, each with its version and, when
there is one, an update button. An add-on someone edited by hand is marked
`local changes` and is not updated over.

NeMo and ONNX audio adapters use isolated Python environments. They are shown
as installed capabilities and can be selected only for compatible tasks and
weight formats. Their environments are deliberately separate from the manager
and vLLM so a speech dependency cannot replace the CUDA stack used elsewhere.

**Engine paths** sits under Engines and names the program used to launch each
installed engine. A change takes effect the next time a model starts.
**Paths** sits under Machine. It holds the two model stores —
**Production models** (core) and **Temporary / benchmark models**, explained in
[Library](library.md#two-places-to-keep-models) — and, below them, the folder
derived from the production store for every text weight format and audio task.
A derived folder that is missing or read-only is marked so. Audio formats used by
different engines are grouped under transcription, voice activity detection
and speaker diarization; engine-specific submodels do not become separate
storage locations. The paths that are choices are picked from a listing of
what is actually on the machine rather than typed — a path typed by hand is a
path with a typo in it, and the failure arrives much later as a screen with no
models on it.

---

[← all documents](../README.md)  ·  [Gateway](gateway.md)  ·  [Writing a request](requests.md)
