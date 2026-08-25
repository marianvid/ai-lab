# Library — what is on disk, and what could be

Every model found on disk, grouped by weight format and task, with what it can
do and whether the set is complete. A repository may expose only a subtree,
which lets the safetensors and native `.nemo` views share `/models/audio/asr`
without listing each other's files. Below it, a search of Hugging Face: pick a
repository, see which of its models this machine can actually run, and download
one whole — never file by file, because four shards of five is a model that
fails to load with an unhelpful message.

![The library](screenshots/library.png)

**Clear** puts a search away and brings back what is on disk. It appears only
once there is something to clear, because a button that does nothing is one
people press once and stop trusting.

A download goes to the folder for its format, decided by the server rather than
chosen. Deleting a model is refused while a configured entry points at it, and
that refusal sends you to Models to remove the entry first — two actions on two
pages, so it is always clear which kind of data is about to disappear.

---

[← all documents](../README.md)  ·  [Models](models.md)  ·  [Audio](audio.md)  ·  [Gateway](gateway.md)
