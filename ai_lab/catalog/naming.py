"""Rules about model file names.

Pure text rules, with no filesystem and no network: what makes a file a shard,
what its base name is, and which files are companions rather than weights.

They live here rather than inside a service because two services need them and
must not import each other. The catalog applies them to files on disk; the
downloader applies the same rules to a listing from Hugging Face, which is why
a model downloads as one set and then appears as one model.
"""

from __future__ import annotations

import re

# "llama-00002-of-00005" -> base "llama", index 2, total 5
SHARD = re.compile(r"^(?P<base>.+)-(?P<index>\d{5})-of-(?P<total>\d{5})$")

# Files that hold weights, and the container they use.
WEIGHT_SUFFIXES = {
    ".gguf": "gguf",
    ".safetensors": "safetensors",
    ".nemo": "nemo",
    ".onnx": "onnx",
    ".pdiparams": "paddleocr",
}

# These repository formats describe how safetensors weights are encoded or
# consumed, not a different file container. They all live in `.safetensors`
# files on disk.
SAFETENSORS_FORMATS = frozenset({
    "safetensors", "fp8", "nvfp4", "awq", "gptq", "comfyui", "mlx",
})

# Files that belong to a model without holding weights.
COMPANION_NAMES = frozenset({
    "config.json", "generation_config.json", "special_tokens_map.json",
    "vocab.json", "merges.txt", "preprocessor_config.json",
    "chat_template.jinja", "model.safetensors.index.json",
    "inference.json", "inference.yml",
})
COMPANION_PREFIXES = ("tokenizer",)

# Weight files that are a part of a model rather than a model of their own.
#
# A vision projector is the clear case. `mmproj-gemma-4-26B-A4B-it-f16.gguf`
# sits beside the model's own GGUF and holds the part that turns a picture into
# something the model can read. It ends in .gguf and it really is weights, but
# llama.cpp is handed it with --mmproj *alongside* the model; on its own it
# loads nothing. Counted as a model it becomes a second entry in the library
# that can never be started — the same fault that used to make
# `model_mtp.safetensors` look like a model, which the directory-is-the-model
# rule fixed for every format except GGUF. GGUF needs saying explicitly,
# because there one file really is one model.
# The separator is part of the rule. Matching a bare "mmproj" would also catch
# a model that happens to begin with those letters, and silently hiding a real
# model is a worse fault than showing a projector.
PART_PREFIXES = ("mmproj-",)
PART_NAMES = frozenset({"mmproj"})


def is_part(name: str) -> bool:
    """Whether a weight file is a component of a model rather than a model."""
    base = stem(name).lower()
    return base in PART_NAMES or base.startswith(PART_PREFIXES)


def is_weight(name: str) -> bool:
    """Whether a file name is model weights, by its suffix."""
    return any(name.lower().endswith(suffix) for suffix in WEIGHT_SUFFIXES)


def is_weight_for_format(name: str, repository_format: str) -> bool:
    """Whether a weight file belongs in a repository's logical format.

    Repositories may intentionally share a directory, as the Hugging Face and
    NeMo ASR repositories do. A plain `is_weight` check would make every
    checkpoint appear once under each repository, labelled with whichever
    format happened to scan it. Match the actual file container as well.
    """
    lowered = name.lower()
    container = next((kind for suffix, kind in WEIGHT_SUFFIXES.items()
                      if lowered.endswith(suffix)), None)
    # ComfyUI bundles may deliberately mix a GGUF diffusion model with
    # safetensors text encoders and VAEs in the same model directory.  The
    # directory is the runnable unit, so both containers belong to it.
    if repository_format == "comfyui":
        return container in {"safetensors", "gguf"}
    if container == "safetensors":
        return repository_format in SAFETENSORS_FORMATS
    return container == repository_format


def is_companion(name: str) -> bool:
    """Whether a file travels with the weights (configuration, tokenizer and the like)."""
    return name in COMPANION_NAMES or name.startswith(COMPANION_PREFIXES)


def stem(name: str) -> str:
    """The file name without its weight suffix."""
    for suffix in WEIGHT_SUFFIXES:
        if name.lower().endswith(suffix):
            return name[: -len(suffix)]
    return name


def split_shard(stem_name: str) -> tuple[str, int, int] | None:
    """Return (base, index, total) for a shard name, or None if it is not one."""
    match = SHARD.match(stem_name)
    if not match:
        return None
    return match["base"], int(match["index"]), int(match["total"])


def base_name(file_name: str) -> str:
    """The model a weight file belongs to, ignoring its shard number."""
    shard = split_shard(stem(file_name))
    return shard[0] if shard else stem(file_name)


def missing_shards(names: list[str]) -> tuple[bool, tuple[str, ...]]:
    """Whether a group of shards is complete, and which ones are absent.

    Every shard states the total in its own name, so a gap is detectable
    without any manifest: five files that all say "of 00005" are complete,
    four are not.
    """
    shards = [shard for shard in (split_shard(stem(name)) for name in names) if shard]
    if not shards:
        return True, ()
    totals = {total for _, _, total in shards}
    if len(totals) != 1:
        return False, ("inconsistent shard numbering",)
    total = totals.pop()
    gaps = sorted(set(range(1, total + 1)) - {index for _, index, _ in shards})
    if not gaps:
        return True, ()
    base = shards[0][0]
    return False, tuple(f"{base}-{index:05d}-of-{total:05d}" for index in gaps)
