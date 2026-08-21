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
WEIGHT_SUFFIXES = {".gguf": "gguf", ".safetensors": "safetensors"}

# Files that belong to a model without holding weights.
COMPANION_NAMES = frozenset({
    "config.json", "generation_config.json", "special_tokens_map.json",
    "vocab.json", "merges.txt", "preprocessor_config.json",
    "chat_template.jinja", "model.safetensors.index.json",
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
    return any(name.lower().endswith(suffix) for suffix in WEIGHT_SUFFIXES)


def is_companion(name: str) -> bool:
    return name in COMPANION_NAMES or name.startswith(COMPANION_PREFIXES)


def stem(name: str) -> str:
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
    parsed = [split_shard(stem(name)) for name in names]
    if not any(parsed):
        return True, ()
    totals = {item[2] for item in parsed if item}
    if len(totals) != 1:
        return False, ("inconsistent shard numbering",)
    total = totals.pop()
    present = {item[1] for item in parsed if item}
    gaps = sorted(set(range(1, total + 1)) - present)
    if not gaps:
        return True, ()
    base = next(item[0] for item in parsed if item)
    return False, tuple(f"{base}-{index:05d}-of-{total:05d}" for index in gaps)
