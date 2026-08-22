"""Deciding which changes matter on this machine.

An engine is built for a dozen kinds of hardware. llama.cpp had 138 changes
waiting on the container, and 26 of them were for Vulkan, SYCL, Metal, OpenCL
and ROCm — none of which exist there. Another 23 were build scripts and 10 were
its own web page. Reading all 138 to find the four about CUDA is not reading,
it is searching.

So the changes are sorted, never dropped. What this machine uses comes first;
the rest is kept behind a count, because a summary that quietly throws things
away is one nobody can trust.

Nothing here is written down by hand about *this* machine. The Mac cares about
Metal and the container cares about CUDA, and both are told which they are by
`Interests`, which is worked out from the accelerator and from what is
configured.
"""

from __future__ import annotations

import re

from ..types import Change, Interests

# What each area is called upstream. Both engines write a prefix at the front
# of a line — "cuda : fix …", "server: add …" — and these turn that prefix into
# a name a person reads.
#
# The order matters: the first pattern that matches wins, so the specific ones
# come before the general. "ggml-cuda" is CUDA, not core.
AREAS: tuple[tuple[str, str], ...] = (
    ("new models",      r"^(model|models|convert-hf)\b"),
    ("CUDA",            r"^(cuda|ggml-cuda|nvidia|blackwell|sm\d+)\b"),
    ("Metal",           r"^(metal|ggml-metal)\b"),
    ("Vulkan",          r"^(vulkan|ggml-vulkan)\b"),
    ("SYCL",            r"^(sycl|ggml-sycl)\b"),
    ("OpenCL",          r"^(opencl|ggml-opencl)\b"),
    ("ROCm",            r"^(hip|rocm|ggml-hip)\b"),
    # Every other piece of silicon somebody has ported this to: Qualcomm's
    # Hexagon, ARM's KleidiAI, Huawei's CANN, Moore Threads' MUSA, the CPU
    # fallback, WebGPU, the network backend. None of them is this machine.
    ("other hardware",  r"^(cann|musa|webgpu|ggml-webgpu|rpc|blas|openblas"
                        r"|cpu|ggml-cpu|hexagon|kleidiai|arm|riscv|loongarch"
                        r"|s390x|powerpc|wasm|zdnn|ggml-zdnn|ggml-blas)\b"),
    ("server",          r"^(server|api|openai|http|webui-server|chat|arg|args"
                        r"|cli|main|tools?)\b"),
    ("pictures",        r"^(mtmd|clip|llava|vision|audio|whisper)\b"),
    ("quantisation",    r"^(quant|quantize|imatrix|gguf|gguf-py|convert"
                        r"|convert_hf_to_gguf)\b"),
    ("core",            r"^(ggml|llama|llama\.cpp|common|graph|kv-cache|memory"
                        r"|vocab|unicode|sampling|spec|speculative|fit|tp"
                        r"|tensor-split|sync|batch|context|threading|opt)\b"),
    ("build",           r"^(ci|cmake|build|make|devops|nix|docker|deps?|vendor"
                        r"|release|version|editorconfig|flake|pre-commit)\b"),
    ("web interface",   r"^(ui|webui)\b"),
    ("documentation",   r"^(docs?|readme|examples?|scripts?|license)\b"),
    ("tests",           r"^(tests?|bench|llama-bench|perplexity|fuzz)\b"),
)

# The prefix is written two ways — "sycl : …" and "[SYCL] …" — and undoing a
# change is written a third, "Revert \"sycl : …\"". All three are the same
# statement about which part of the engine is involved, so the text is put in
# one shape before any pattern is tried.
BRACKETED = re.compile(r"^\[([^\]]+)\]\s*")
REVERTED = re.compile(r'^Revert\s+"(.*)"\s*(?:\(#\d+\))?\s*$', re.I)

# Areas that belong to a piece of hardware. An area not in here — the server,
# new models, quantisation — is about the engine itself and matters wherever it
# runs.
HARDWARE = {
    "CUDA": "cuda",
    "Metal": "metal",
    "Vulkan": "vulkan",
    "SYCL": "sycl",
    "OpenCL": "opencl",
    "ROCm": "rocm",
    "other hardware": "other",
}

# Areas nobody running an engine needs to read about. They are still counted
# and still there to open; they are simply never the answer to "what would
# change for me".
BACKSTAGE = frozenset({"build", "web interface", "documentation", "tests"})

UNSORTED = "unsorted"


def area_of(title: str) -> str:
    """Which part of the engine a line is about, from the prefix it carries."""
    text = _plain(title)
    for name, pattern in AREAS:
        if re.match(pattern, text, re.I):
            return name
    return UNSORTED


def _plain(title: str) -> str:
    """One shape for the three ways a prefix gets written.

    Undoing a change is about the same part of the engine as the change was,
    so a revert is classified by what it reverts. Nested reverts happen — a
    revert of a revert — so this unwraps until it stops changing.
    """
    text = (title or "").strip()
    for _ in range(4):                       # a bound, not an expectation
        before = text
        text = BRACKETED.sub(lambda match: f"{match.group(1)}: ", text).strip()
        undone = REVERTED.match(text)
        if undone:
            text = undone.group(1).strip()
        if text == before:
            break
    return text


def matters(area: str, interests: Interests) -> bool:
    """Would this change anything for the machine described by `interests`?"""
    if area in BACKSTAGE:
        return False
    hardware = HARDWARE.get(area)
    if hardware is not None:
        # Somebody else's card. The one exception is the accelerator this
        # machine actually has.
        return hardware == interests.accelerator_kind
    if area == "pictures":
        # Only worth reading if some configured entry can use them.
        return interests.pictures
    # Anything left over is unsorted, or is about the engine itself: the
    # server, new model architectures, quantisation, the core. Those matter
    # wherever it runs. Unsorted is included on purpose — a change nobody has
    # classified is more likely to be missed than to be noise.
    return True


def sift(changes: list[Change], interests: Interests
         ) -> tuple[tuple[Change, ...], tuple[Change, ...]]:
    """Split changes into the ones this machine cares about, and the rest.

    Order is preserved within each half, because both engines list newest
    first and that is the order somebody reads them in.
    """
    yours, others = [], []
    for change in changes:
        (yours if matters(change.area, interests) else others).append(change)
    return tuple(yours), tuple(others)


def counted(changes: tuple[Change, ...]) -> dict[str, int]:
    """How many in each area, for a line that says what is in there."""
    tally: dict[str, int] = {}
    for change in changes:
        tally[change.area] = tally.get(change.area, 0) + 1
    return dict(sorted(tally.items(), key=lambda pair: (-pair[1], pair[0])))
