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
installed; `Update status unavailable` means it could not yet be read. When a
new version exists it is named beside the Update button. No button here updates
anything directly: it opens what the update would bring, and the real Update
is at the foot of that. See
[Updating an engine](engines.md).

NeMo and ONNX audio adapters use isolated Python environments. They are shown
as installed capabilities and can be selected only for compatible tasks and
weight formats. Their environments are deliberately separate from the manager
and vLLM so a speech dependency cannot replace the CUDA stack used elsewhere.

**Paths** is every path this installation depends on: one models root with a
folder per weight format under it, and each engine's program. All of them are
picked from a listing of what is actually on the machine rather than typed — a
path typed by hand is a path with a typo in it, and the failure arrives much
later as a screen with no models on it.

---

[← all documents](../README.md)  ·  [Gateway](gateway.md)  ·  [Writing a request](requests.md)
