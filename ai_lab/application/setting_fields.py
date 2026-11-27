"""Every setting kept in the configuration file, as a field on the Settings page.

The owner's rule (2026-11-27): no setting lives only in a file. Each one here
carries what the page needs to show it and explain it — its label, unit,
default, the range it may take, a plain sentence on what it means and what too
small or too large does, and whether it takes effect only after a restart —
and the place it is kept (`key`, dotted: `limits.runtime.load_s` is
`config.json`'s `limits` → `runtime` → `load_s`).

A change is written into a copy of the configuration and checked by the parser
that owns its section (`config.policy`) before it is saved, so the page and the
start of AI-Lab refuse the same values with the same words.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass

from ..config import policy
from ..config.store import Config, ConfigStore

# The groups the page shows, in order: the section title of each.
GROUPS = {
    "loading": "Loading models",
    "updates": "Engine updates",
    "embeddings": "Embeddings lane",
    "clients": "Client caps",
}
RESTART = "Takes effect when AI-Lab is next started."


@dataclass(frozen=True)
class Field:
    """One setting: where it is kept, how the page shows it, and what it means."""

    key: str
    group: str
    label: str
    unit: str
    default: float | int | str
    low: float | None
    high: float | None
    help: str
    kind: str = "number"            # "number", "integer" or "text"
    restart: bool = True


def _seconds(key: str, group: str, label: str, default: float, help_text: str) -> Field:
    """A wait in seconds, from one second to a week (`policy.LONGEST_S`), as its parser takes."""
    return Field(key, group, label, "s", default, 1, policy.LONGEST_S, help_text)


FIELDS: tuple[Field, ...] = (
    _seconds("limits.runtime.start_s", "loading", "Process start", policy.START_S,
             "How long a model's engine process may take to appear after it is started. "
             "Too small: an engine that starts slowly is given up on and the load fails. "
             "Too large: a process that never appears is waited on for that long."),
    _seconds("limits.runtime.load_s", "loading", "Model load", policy.LOAD_S,
             "How long the engine may take to answer once started: reading the weights, "
             "a first compile. Too small: a large model, a slow disk or a first compile "
             "fails its load. Too large: a load that is stuck holds the card that long."),
    _seconds("limits.runtime.unload_s", "loading", "Model unload", policy.UNLOAD_S,
             "How long an engine may take to exit when stopped, and again for its memory "
             "to come back. Too small: a stop is reported as failed while memory is still "
             "being freed. Too large: a stuck stop holds the next load that long."),
    Field("limits.first_port", "loading", "First port offered", "", policy.FIRST_PORT,
          1, 65535, "The first port suggested when a model entry is added; the next free "
          "one above it is offered. Only a suggestion: each entry's port can be changed. "
          "Pick a range nothing else on this machine uses.", kind="integer", restart=False),
    _seconds("limits.builds.install_s", "updates", "Install step", policy.INSTALL_S,
             "How long one install command (a package install, a ComfyUI step) may run. "
             "Too small: a slow download is cut and the install fails, the old version "
             "kept. Too large: a hung install shows \u201cinstalling\u201d that long."),
    _seconds("limits.builds.build_s", "updates", "Compile step", policy.BUILD_S,
             "How long one compile step of an engine built from source (llama.cpp) may run. "
             "Too small: a full rebuild is cut and the old build kept. Too large: a hung "
             "compile shows \u201cbuilding\u201d that long."),
    _seconds("limits.builds.check_every_s", "updates", "Check for updates every",
             policy.UPDATE_CHECK_EVERY_S,
             "How often upstream is asked whether a newer version exists, so the page "
             "already knows. Too small: upstream is asked needlessly often. Too large: a new "
             "version is noticed late."),
    _seconds("limits.builds.builds_first_check_s", "updates", "First check, source builds",
             policy.BUILDS_FIRST_CHECK_S,
             "How long after start the source builds first ask upstream, so starting is not "
             "held up by the network. Kept apart from the installs' first check."),
    _seconds("limits.builds.installs_first_check_s", "updates", "First check, installs",
             policy.INSTALLS_FIRST_CHECK_S,
             "How long after start the package installs first ask the index. Kept apart "
             "from the source builds' first check, so the two do not ask at once."),
    Field("limits.builds.system_path", "updates", "Search path for installs", "",
          policy.SYSTEM_PATH, None, None,
          "Where the programs a new engine environment is built with (python, git, uv, "
          "compilers) are looked for: absolute folders joined by \u201c:\u201d. Wrong: an "
          "install or update cannot find its tools and fails, the old version kept.",
          kind="text"),
    _seconds("limits.embeddings.start_s", "embeddings", "Server start",
             policy.EMBEDDINGS_START_S,
             "How long an embeddings model's server may take to answer after it starts. Too "
             "small: a model loading from a slow disk fails its first call. Too large: a "
             "server that never comes up holds its first call that long."),
    _seconds("limits.embeddings.stop_s", "embeddings", "Server stop", policy.EMBEDDINGS_STOP_S,
             "How long that server may take to exit when asked before it is killed. Too "
             "small: it is killed mid-cleanup. Too large: stopping waits that long."),
    _seconds("limits.embeddings.drain_s", "embeddings", "Leave the card within",
             policy.EMBEDDINGS_DRAIN_S,
             "When a card model needs the card, how long the embeddings calls already on it "
             "may take to finish before their server is stopped anyway. Too small: those "
             "calls fail. Too large: the card model's load waits that long."),
    _seconds("limits.embeddings.hold_s", "embeddings", "Stay off the card for",
             policy.EMBEDDINGS_HOLD_S,
             "A safety net: if a card load never says it is done, the card is offered to the "
             "embeddings again after this long. Too small: embeddings may return while the "
             "load still needs the room. Too large: they stay on the processor that long."),
    _seconds("limits.embeddings.call_s", "embeddings", "One call", policy.EMBEDDINGS_CALL_S,
             "How long one embeddings call to its server may take, the queue wait not "
             "counted. Too small: a large batch of long texts fails. Too large: a stuck call "
             "holds its place that long."),
    Field("limits.clients.chars_per_token", "clients", "Characters per token", "",
          policy.CHARS_PER_TOKEN, 0.5, 20,
          "A capped client's prompt is counted as its characters divided by this, before "
          "the model sees it. Smaller counts more tokens (safer for the pool, refuses "
          "sooner); larger counts fewer (lets more in, may overfill the pool). English runs "
          "about 4, code about 3."),
    Field("limits.clients.retry_after_s", "clients", "Retry after", "s", policy.RETRY_AFTER_S,
          1, 3600, "How long a client over its cap is told to wait (the Retry-After header). "
          "Too small: it retries and is refused again. Too large: it waits longer than "
          "needed.", kind="integer"),
)
BY_KEY = {field.key: field for field in FIELDS}


def _read(config: Config, key: str):
    """The value saved at `key`, or None when the file does not set it."""
    section, *path = key.split(".")
    value = getattr(config, section)
    for part in path:
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def _write(config: Config, key: str, value) -> None:
    """Set `key` in the configuration, creating the sections on the way."""
    section, *path = key.split(".")
    holder = getattr(config, section)
    for part in path[:-1]:
        holder = holder.setdefault(part, {})
    holder[path[-1]] = value


def _checked(field: Field, value):
    """The value as the field takes it; ValueError naming the field otherwise."""
    if field.kind == "text":
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field.label} must be text")
        return value.strip()
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field.label} must be a number")
    if field.kind == "integer":
        if value != int(value):
            raise ValueError(f"{field.label} must be a whole number")
        value = int(value)
    return value


class SettingFields:
    """The fields, with their saved values, and the change of some of them."""

    def __init__(self, store: ConfigStore, changed: Callable[[str], None]) -> None:
        self.store = store
        self._changed = changed

    def view(self) -> dict:
        """Every field with its value now (its default when the file sets none), by group."""
        config = self.store.load()
        fields = []
        for field in FIELDS:
            saved = _read(config, field.key)
            fields.append({**asdict(field), "value": field.default if saved is None else saved,
                           "saved": saved is not None,
                           "restart_note": RESTART if field.restart else ""})
        return {"groups": GROUPS, "fields": fields}

    def update(self, changes: dict) -> dict:
        """Save some fields, each checked by the parser that owns its section."""
        if not isinstance(changes, dict) or not changes:
            raise ValueError("Send the fields to change, by key")
        unknown = sorted(set(changes) - set(BY_KEY))
        if unknown:
            raise ValueError("Unknown settings: " + ", ".join(unknown))
        with self.store.mutate() as config:
            for key, value in changes.items():
                _write(config, key, _checked(BY_KEY[key], value))
            # The section's own parser, as at start: ValueError leaves the file as it was.
            policy.MachineLimits.from_mapping(config.limits)
        self._changed("settings")
        return self.view()
