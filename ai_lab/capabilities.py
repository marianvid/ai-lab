"""What a model can do, worked out from its own files.

Two things are worth knowing before a model is started: whether it can read
pictures, and whether it can call tools. Neither is a setting somebody chose —
they are properties of the weights — so they are read rather than configured,
and an entry can only take them away.

The evidence differs by format, and in both cases it is what the engine itself
looks at:

**Pictures.** A GGUF model is multimodal when a projector sits beside it —
`mmproj-<model>-f16.gguf`, the part that turns an image into something the
model can read. Everything else is a directory, and there the model's own
`config.json` says: a `vision_config` and an image token mean it has the
pieces.

**Tools.** Both engines decide from the chat template, so this does too: a
template that mentions tools is a model that has been taught to ask for them.
For a directory that template is a small file beside the weights. For GGUF it
is inside the weights, in the metadata at the head of the file — which is the
expensive part and the reason for the cache below.

Nothing here starts anything or asks a running engine. It reads files, so it
works for a model that has never been loaded, which is when somebody choosing
one wants to know.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

IMAGES = "images"
TOOLS = "tools"

# What a chat template says when it has been taught to ask for tools. Both
# engines look for the same thing, in the same place, so this is not a guess
# about the model — it is the question the engine will ask.
TOOL_MARKER = "tools"

# Where a directory-shaped model keeps its chat template. The first is the
# current convention and the second is where it used to live.
TEMPLATE_FILE = "chat_template.jinja"
TOKENIZER_CONFIG = "tokenizer_config.json"

# GGUF's metadata sits at the head of the file, but the chat template is behind
# the tokenizer's vocabulary — measured on this machine, between six and
# sixteen megabytes in, and between 90 and 245 milliseconds to reach. With a
# dozen models that is two seconds on every scan of the library, so an answer
# is kept until the file changes.
_GGUF_MAGIC = b"GGUF"
_FIXED_WIDTHS = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1,
                 10: 8, 11: 8, 12: 8}
_STRING, _ARRAY = 8, 9
_CHAT_TEMPLATE_KEY = "tokenizer.chat_template"

FILE_NAME = "model-capabilities.json"


class Known:
    """What has already been read, so a file is opened once and not again.

    Kept on disk beside the rest of the state, because the answer costs a
    quarter of a second for a GGUF model and does not change while the file
    does not. The key is the path with the size and the modification time, so a
    model replaced under the same name is read again rather than believed.

    Every failure here is quiet. Losing this costs a re-read; refusing to serve
    a page over it would cost more.
    """

    def __init__(self, directory: "Path | None" = None) -> None:
        self.path = Path(directory) / FILE_NAME if directory else None
        self._known: dict[str, list[str]] = self._load()

    def of(self, entrypoint: str, is_gguf: bool,
           companions: list[str]) -> frozenset[str]:
        """What this model can do, reading its files only the first time.

        `entrypoint` is what the engine is handed — the first weight file for
        GGUF, the directory for everything else. `companions` are the file
        names sitting beside it.

        Never raises: a model whose files cannot be read is reported as able to
        do nothing in particular, which is what showing no icon means anyway.
        """
        path = Path(entrypoint)
        key = self._key(path, is_gguf)
        if key is None:
            return frozenset()
        remembered = self._known.get(key)
        if remembered is not None:
            return frozenset(remembered)
        try:
            found = _read(path, is_gguf, companions)
        except Exception:
            found = frozenset()
        self._known[key] = sorted(found)
        self._save()
        return found

    def forget(self) -> None:
        """Drop everything read. For a library that was replaced, and for tests."""
        self._known.clear()
        self._save()

    # -- keeping it ---------------------------------------------------------

    @staticmethod
    def _key(path: Path, is_gguf: bool) -> str | None:
        try:
            stamp = path.stat()
        except OSError:
            return None
        # A directory's own mtime says nothing about the files in it, so for
        # those the key is the path alone and a changed model is a changed
        # path. GGUF is one file, and there the size and the time are exactly
        # the question.
        if not is_gguf:
            return f"dir:{path}"
        return f"gguf:{path}:{stamp.st_size}:{stamp.st_mtime_ns}"

    def _load(self) -> dict:
        if self.path is None:
            return {}
        try:
            stored = json.loads(self.path.read_text())
        except (OSError, ValueError):
            return {}
        return stored if isinstance(stored, dict) else {}

    def _save(self) -> None:
        if self.path is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix(".tmp")
            temporary.write_text(json.dumps(self._known, indent=2, sort_keys=True))
            temporary.replace(self.path)
        except OSError:
            pass


# -- the two shapes a model comes in ---------------------------------------

def _read(path: Path, is_gguf: bool, companions: list[str]) -> frozenset[str]:
    if is_gguf:
        found = set()
        if any(name.lower().startswith("mmproj") for name in companions):
            found.add(IMAGES)
        if _mentions_tools(_gguf_chat_template(path)):
            found.add(TOOLS)
        return frozenset(found)

    found = set()
    if _directory_sees_images(path):
        found.add(IMAGES)
    if _mentions_tools(_directory_chat_template(path)):
        found.add(TOOLS)
    return frozenset(found)


def _mentions_tools(template: str) -> bool:
    return TOOL_MARKER in (template or "").lower()


def _directory_sees_images(directory: Path) -> bool:
    """A vision section and an image token, in the model's own config."""
    try:
        config = json.loads((directory / "config.json").read_text())
    except (OSError, ValueError):
        return False
    if not isinstance(config, dict):
        return False
    return bool(config.get("vision_config")) and "image_token_id" in config


def _directory_chat_template(directory: Path) -> str:
    try:
        return (directory / TEMPLATE_FILE).read_text()
    except OSError:
        pass
    try:
        config = json.loads((directory / TOKENIZER_CONFIG).read_text())
    except (OSError, ValueError):
        return ""
    template = config.get("chat_template") if isinstance(config, dict) else ""
    # Some models carry several, as a list of {name, template}.
    if isinstance(template, list):
        return " ".join(str(item.get("template", "")) for item in template
                        if isinstance(item, dict))
    return template if isinstance(template, str) else ""


# -- reading the head of a GGUF file ---------------------------------------

def _gguf_chat_template(path: Path) -> str:
    """The chat template out of a GGUF file's metadata, or "".

    Walks the key-value pairs at the head of the file until it finds the
    template, skipping over the rest — which includes the vocabulary, and is
    why this is not free.
    """
    with path.open("rb") as handle:
        magic, _version, _tensors, pairs = struct.unpack("<4sIQQ",
                                                         handle.read(24))
        if magic != _GGUF_MAGIC:
            return ""

        def number(fmt: str, width: int):
            return struct.unpack(fmt, handle.read(width))[0]

        def text() -> str:
            return handle.read(number("<Q", 8)).decode("utf-8", "replace")

        def step_over(kind: int) -> None:
            if kind == _STRING:
                text()
            elif kind == _ARRAY:
                inner = number("<I", 4)
                for _ in range(number("<Q", 8)):
                    step_over(inner)
            else:
                handle.read(_FIXED_WIDTHS[kind])

        for _ in range(pairs):
            key = text()
            kind = number("<I", 4)
            if key == _CHAT_TEMPLATE_KEY:
                return text()
            step_over(kind)
    return ""
