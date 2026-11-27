"""Every setting of the configuration file, with the page that shows it.

The owner's rule (2026-11-27): no setting lives only in a file, and where it is
not certain how a setting should be changed, it is shown read-only ("Hai sa
fie read only acolo unde nu avem certitudine. macar sa stiu despre ele").

`SCHEMA` names, for every key at the top of `config.json`, the page that
shows it and whether it is changed there; `tests/test_file_settings.py` fails
when a key of `Config` is missing from it, so a setting added later cannot
hide. `READ_ONLY` is what this module itself shows: the record-shaped
sections, each with what it is, what it is for, where it is changed and
whether a restart is needed. A secret (a token, a key, a password) is never
shown: only that it is set.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from ..config.store import Config, ConfigStore

# How a key is changed: on its page ("editable"), only in the file and shown
# on its page ("read-only"), or field by field on the Settings page
# (`setting_fields.py`) and the Gateway page ("fields").
EDITABLE, READ_ONLY_KIND, FIELDS = "editable", "read-only", "fields"

# Every top-level key of `Config`: the page that shows it, and how.
SCHEMA: dict[str, tuple[str, str]] = {
    "schema_version": ("Settings → read-only sections at the foot of the left column (File format version)", READ_ONLY_KIND),
    "title": ("Settings → This AI-Lab", EDITABLE),
    "host": ("Settings → This AI-Lab", READ_ONLY_KIND),
    "port": ("Settings → This AI-Lab", READ_ONLY_KIND),
    "repositories": ("Settings → model paths; Library", EDITABLE),
    "instances": ("Models", EDITABLE),
    "engines": ("Settings → Engines (binary editable); the rest in read-only sections at the foot of the left column", READ_ONLY_KIND),
    "gateway": ("Gateway → Limits; Settings → Uploaded images", FIELDS),
    "limits": ("Settings (Loading models, Engine updates, Embeddings lane, Client caps)",
               FIELDS),
    "memory": ("Settings → Machine", EDITABLE),
    "storage": ("Storage", READ_ONLY_KIND),
    "images": ("Library", READ_ONLY_KIND),
    "downloads": ("Library", READ_ONLY_KIND),
    "model_notes": ("Library", READ_ONLY_KIND),
    "providers": ("Lanes", READ_ONLY_KIND),
    "cpu_models": ("Lanes", READ_ONLY_KIND),
    "gpu_health": ("Settings → GPU health (channel read-only)", FIELDS),
    "models_root": ("Settings → model paths", EDITABLE),
    "model_roots": ("Settings → model paths", EDITABLE),
    "download_root": ("Settings → model paths (a root's \u201cdownload default\u201d)", EDITABLE),
    "extra_fields": ("Settings → read-only sections at the foot of the left column", READ_ONLY_KIND),
}

RESTART = "Read when AI-Lab starts: a change in the file takes effect at the next start."
# A name (of a key, an environment variable, a command-line flag) whose value
# may be a secret: shown as set, never as its value. `key` and `pass` only as
# a whole word or a last part (`private_key`, `OPENAI_KEY`), not `keep`.
SECRET = re.compile(r"token|secret|passw|(^|[_\-.])pass($|[_\-.])|api[_\-]?key|credential|auth"
                    r"|bearer|cookie|session|signature|private|(^|[_\-.])key$", re.I)
# Credentials inside a URL, whatever their form (`https://user:PASS@host`,
# `https://TOKEN@host`, a Git remote's), and a bearer header value.
URL_USERINFO = re.compile(r"(\b[a-z][a-z0-9+.\-]*://)([^/@\s]+)@", re.I)
BEARER = re.compile(r"\bbearer\s+[^\s,;]+", re.I)
# A URL's query parameter (`?api_key=V`, `&token=V`), a `NAME=VALUE` word
# (`OPENAI_API_KEY=V`, an environment item or a command's prefix), and a header
# line (`X-Api-Key: V`): hidden when the name is a secret's.
URL_QUERY = re.compile(r"([?&])([^=&#\s]+)=([^&#\s]*)")
ASSIGNMENT = re.compile(r"^([A-Za-z_][\w\-.]*)=(.*)$", re.S)
HEADER = re.compile(r"^([A-Za-z][\w\-]*)\s*:\s*(\S.*)$", re.S)
# A command-line flag: `--api-key`, `-t`, `--token=VALUE`.
FLAG = re.compile(r"^-{1,2}([A-Za-z][\w\-.]*)(=(.*))?$", re.S)


@dataclass(frozen=True)
class Shown:
    """One read-only section: where it is kept, which page shows it, what it means."""

    key: str
    page: str            # the page's name in the menu: "library", "storage", "lanes", "settings"
    title: str
    help: str
    restart: str = RESTART


READ_ONLY = (
    Shown("schema_version", "settings", "File format version",
          "The version of this file's layout. AI-Lab writes it; it is not a choice. "
          "Changed only by AI-Lab itself when the layout changes."),
    Shown("extra_fields", "settings", "Keys this version does not know",
          "Keys at the top of config.json that this AI-Lab does not read, kept as they "
          "are so a newer AI-Lab still finds them. Empty normally."),
    Shown("engines", "settings", "Engine settings beyond the program",
          "Per engine, what it is configured with besides its program (shown editable "
          "above): per-model options (model_options: a workflow, a voice, a memory "
          "reservation), its locations (folders, programs), preset workflows. Changed in "
          "config.json under engines.<engine>."),
    Shown("images", "library", "Image profiles",
          "The named image profiles a client may ask for (POST /v1/images/...): each "
          "names a model and a ComfyUI workflow file under workflow_root. Changed in "
          "config.json under images (workflow_root, profiles)."),
    Shown("downloads", "library", "Download bundles",
          "Models fetched together as one (a model and the files it needs beside it). "
          "Changed in config.json under downloads.bundles."),
    Shown("model_notes", "library", "Notes on models",
          "Per model name: a short line on why to choose it, and a longer one for its "
          "tooltip in the Library. Changed in config.json under model_notes; read on "
          "each page load.", restart="Read on each page load: no restart."),
    Shown("storage", "storage", "Reclaimable folders",
          "The folders outside model storage that may be emptied from the Storage page "
          "(caches): each with its name, path and kind. Changed in config.json under "
          "storage.reclaimable."),
    Shown("providers", "lanes", "Subscription models",
          "The vendors answered through their own command-line tools (Claude Code, "
          "Codex): their models, how many at once, their usage limits. Changed in "
          "config.json under providers. Its secrets are not shown."),
    Shown("cpu_models", "lanes", "Models on the processor",
          "The embeddings models run from RAM: the llama.cpp server, each model's file, "
          "port and limits. Changed in config.json under cpu_models."),
)


def _set(value) -> str:
    return "(set)" if value else "(empty)"


def masked(value, key: str = ""):
    """`value` with every secret replaced by "(set)" (or "(empty)"), at any depth.

    Under a key that names a secret everything is hidden, whatever its shape
    (text, a list, an object); only a number there is shown as it is
    (`token_warn_after_days`). Elsewhere, a secret is also found by its value:
    the word after a secret's flag in a list of arguments (`--api-key VALUE`,
    `--token=VALUE`), the password inside a URL, a `Bearer` header.
    """
    if key and SECRET.search(key):
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return value
        return _set(value)
    if isinstance(value, dict):
        return {name: masked(item, str(name)) for name, item in value.items()}
    if isinstance(value, list):
        return _masked_arguments(value)
    if isinstance(value, str):
        return _masked_text(value)
    return value


def _hidden_userinfo(match: re.Match) -> str:
    scheme, userinfo = match.group(1), match.group(2)
    user, colon, _secret = userinfo.partition(":")
    return f"{scheme}{user}:(set)@" if colon else f"{scheme}(set)@"


def _hidden_query(match: re.Match) -> str:
    mark, name, secret = match.group(1), match.group(2), match.group(3)
    return f"{mark}{name}={_set(secret)}" if SECRET.search(name) else match.group(0)


def _masked_text(value: str) -> str:
    """A text with its secrets hidden: a bearer value anywhere, a URL's
    credentials and secret query values, a secret header line, and in a
    command line written as one string, the word after a secret's flag and a
    secret's `NAME=VALUE`."""
    header = HEADER.match(value)
    if header and SECRET.search(header.group(1)):
        return f"{header.group(1)}: (set)"
    value = BEARER.sub("Bearer (set)", value)
    value = URL_USERINFO.sub(_hidden_userinfo, value)
    value = URL_QUERY.sub(_hidden_query, value)
    parts = re.split(r"(\s+)", value)
    if len(parts) == 1:
        assignment = ASSIGNMENT.match(value)
        if assignment and SECRET.search(assignment.group(1)):
            return f"{assignment.group(1)}={_set(assignment.group(2))}"
        return value
    parts[0::2] = _masked_arguments(parts[0::2])
    return "".join(parts)


def _masked_arguments(items: list) -> list:
    """A list, each item masked; after a secret's flag, the next item hidden too."""
    shown: list = []
    hide_next = False
    for item in items:
        if hide_next:
            shown.append(_set(item))
            hide_next = False
            continue
        flag = FLAG.match(item) if isinstance(item, str) else None
        if flag and SECRET.search(flag.group(1)):
            if flag.group(2) is not None:
                shown.append(f"{item.split('=', 1)[0]}={_set(flag.group(3))}")
            else:
                shown.append(item)
                hide_next = True
            continue
        shown.append(masked(item))
    return shown


def _value(config: Config, shown: Shown):
    value = getattr(config, shown.key)
    if shown.key == "engines":
        # The program is shown, and changed, by the Engines section itself.
        value = {engine: {name: item for name, item in settings.items() if name != "binary"}
                 for engine, settings in (value or {}).items()}
    return masked(asdict(value) if hasattr(value, "__dataclass_fields__") else value)


class FileSettings:
    """The read-only sections of one page, with their values as saved."""

    def __init__(self, store: ConfigStore) -> None:
        self.store = store

    def view(self, page: str | None = None) -> dict:
        """The read-only sections for `page` (every page's when None)."""
        config = self.store.load()
        return {"sections": [{**asdict(item), "value": _value(config, item)}
                             for item in READ_ONLY if page in (None, item.page)]}

    @staticmethod
    def schema() -> list[dict]:
        """Every top-level key: the page that shows it and how it is changed."""
        return [{"key": key, "page": page, "how": how} for key, (page, how) in SCHEMA.items()]
