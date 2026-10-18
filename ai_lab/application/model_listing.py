"""The model library as the pages list it: every model on disk, with its notes.

Used by the Models page and by the form for a new entry, which offers only the
models an engine can load. Reads the configuration and the catalog; changes
nothing.
"""

from __future__ import annotations

from ..catalog.core import Catalog
from ..config.store import Config, ConfigStore


class ModelListing:
    """One row per model on disk, optionally only those one engine can load."""

    def __init__(self, store: ConfigStore, catalog: Catalog, engines) -> None:
        self.store = store
        self.catalog = catalog
        self.engines = engines

    def models(self, engine_id: str | None = None) -> list[dict]:
        """Every model on disk, optionally only those an engine can load."""
        config = self.store.load()
        models = self.catalog.scan(config.repositories)
        if engine_id:
            models = self._loadable(models, engine_id)
        return [self._row(item, config) for item in models]

    def _loadable(self, models: list, engine_id: str) -> list:
        """Only the models this engine reads: its formats, its tasks, its own test."""
        engine = self.engines.get(engine_id)
        formats, tasks = engine.formats(), engine.tasks()
        supports = getattr(engine, "supports", None)
        return [item for item in models
                if item.format in formats and item.task in tasks
                and (supports is None or supports(item))]

    def _row(self, item, config: Config) -> dict:
        row = model_row(item)
        row.update(_note(config.model_notes.get(item.id, config.model_notes.get(item.name, {}))))
        try:
            row["storage_tier"] = config.repository(item.id.split("/", 1)[0]).root_id
        except KeyError:
            # The prefix always comes from a configured repository's own
            # id (see `Catalog.scan`), so this cannot happen today — but
            # the listing itself must not fail if it ever does.
            row["storage_tier"] = ""
        return row


def model_row(model) -> dict:
    """The fields of one model every listing shows."""
    return {"id": model.id, "name": model.name, "format": model.format.value,
            "task": model.task.value,
            "entrypoint": model.entrypoint, "size_bytes": model.size_bytes,
            "file_count": len(model.files), "complete": model.complete,
            "missing": list(model.missing),
            "capabilities": sorted(model.capabilities)}


def _note(note) -> dict:
    """A note from the configuration as `summary` and `description`.

    A note is either one sentence, used for both, or a short and a long form.
    """
    if isinstance(note, str):
        note = {"short": note, "detail": note}
    if not isinstance(note, dict):
        return {}
    short = str(note.get("short", "")).strip()
    detail = str(note.get("detail", short)).strip()
    fields = {}
    if short:
        fields["summary"] = short
    if detail:
        fields["description"] = detail
    return fields
