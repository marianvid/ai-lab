# Package-installed engine versions

Installing an engine that arrives as packages, beside the one that works.

vLLM is not compiled here. It is 382 packages and 7.7 GB in a Python virtual
environment — a folder holding its own copy of Python and everything that
version needs, so two of them can sit side by side without touching each other.

The same mechanism serves every engine whose configuration has a
`source.package` entry. In `config.example.json` those are vLLM, NeMo, the
Silero ONNX adapter, pyannote.audio and PaddleOCR. ComfyUI is a git
application rather than a package and has its own, similar path; see
[Updating an engine](engines.md#comfyui-core-and-custom-nodes).

**Nothing is ever installed over what works.** A new version is built in a new
folder, checked that it actually starts, and only then does the engine start
pointing at it. The old folder stays exactly as it was, so going back is
instant and is the same act as going forward.

That is not a nicety here. The vLLM installed on this machine when this was
written cannot be reinstalled: its wheel is no longer in the local cache, and
the index it came from is recorded nowhere — a nightly build with a git hash in
its name, and those indexes are pruned. An update in place would have been
irreversible.

What "checked that it actually starts" means, step by step
(`package_install.py`):

1. `uv` (a fast Python package installer) creates the environment in a folder
   named after the version. It is created at its final path, because an
   environment's launcher scripts hold that absolute path.
2. The pinned package and any configured companion packages are installed.
3. The installed version must be the one asked for.
4. The new environment's Python must import the configured modules, meet any
   configured minimum versions of other packages and, when `requires_cuda` is
   set, see a working CUDA card.

If any step fails, the half-made folder is deleted. The folder in use was
never written to. Installing, and switching versions, are refused while any
model is loaded. Deleting a version is refused only for the one in use.

AI-Lab asks the package index for the newest release by itself: first about 25
seconds after the manager starts, then once an hour (`installs.py`).

How the engine finds the right one: it is launched through a fixed path, a link
called `current` that points at whichever folder is in use. Swapping versions
means repointing that link, which happens in one step — there is no moment when
it points at nothing.

Two folders is the steady state, about 16 GB: the one in use and the one before
it. It does not grow on its own, and it is never tidied automatically — the old
one *is* the way back, so it goes when somebody decides it can, not when a rule
decides for them.
