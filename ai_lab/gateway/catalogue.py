"""What the gateway tells a client about its models, and which name means which.

A base class of `Gateway` (a mixin): the listing, one model's description, the
one name each entry answers to, and which request shapes its engine answers.
Nothing here touches the card or the queue.
"""

from __future__ import annotations

from ..engines.base import withheld
from .errors import NotConfiguredError, ShapeNotServedError
from ..operations import Operations
from ..types import Task


class GatewayCatalogue:
    """Names, listings and shapes, read from the configuration and the engines."""

    operations: Operations

    def catalogue(self) -> list[dict]:
        """Every configured entry, loaded or not, with the names it answers to.

        Entries that are not loaded are listed too. A client is meant to be able
        to ask for one of them — that is the whole point.
        """
        return self._rows(self.operations.instances())

    def describe(self, wanted: str) -> dict:
        """Everything a client needs to know about one model before using it.

        The listing row, plus the settings the model will start with and, for
        speech and music, the fields a request may carry
        (`speech_form`, `music_form`).
        """
        instances = self.operations.instances()
        instance = self.resolve(wanted, instances)
        row = self._rows([instance])[0]
        row["params"] = instance.get("params", {})
        for form in ("speech_form", "music_form"):
            if form in instance:
                row[form] = instance[form]
        return row

    def _rows(self, instances: list[dict]) -> list[dict]:
        """One listing row per entry already read."""
        return [{
            "id": instance["id"],
            "model_id": instance["model_id"],
            "engine": instance["engine"],
            "port": instance["port"],
            "loaded": bool(instance["running"]),
            "ready": bool(instance["ready"]),

            # Which shapes of request this one answers. A client that speaks
            # only one of them can tell from the listing which models are open
            # to it, instead of finding out by being refused.
            "shapes": self._shapes(instance),

            # What kind of work it does, and what it can do beyond plain
            # text: read pictures ("images"), ask for tool calls ("tools").
            # Without these a client has to try a model to find out.
            "task": instance.get("task", Task.TEXT_GENERATION.value),
            "capabilities": self._capabilities(instance),
        } for instance in instances]

    def _capabilities(self, instance: dict) -> list[str]:
        """What the model's own files say it can do, minus what the entry turns off.

        Read from the weights (see `capabilities.py`), not configured. The
        engine can then take things away — vLLM started "text only", or
        mlx-lm, which never reads pictures — so it is asked what it withholds.
        A model whose files cannot be read reports nothing rather than failing
        the whole listing.
        """
        try:
            able = set(self.operations.model_for(instance["id"]).capabilities)
        except Exception:
            return []
        try:
            engine = self.operations.engines.get(instance["engine"])
        except KeyError:
            return sorted(able)
        return sorted(able - withheld(engine, instance.get("params")))

    def _shapes(self, instance: dict) -> list[str]:
        try:
            task = Task(instance.get("task", Task.TEXT_GENERATION.value))
            return list(self.operations.engines.get(instance["engine"]).api_paths(task))
        except KeyError:
            return []

    @staticmethod
    def _engine_name(instance: dict) -> str:
        """What this entry's engine calls its own model.

        The last segment of the model path, which is the name the engine is
        started with and the only one it will answer to.
        """
        model_id = instance.get("model_id") or ""
        return model_id.rsplit("/", 1)[-1]

    def resolve(self, wanted: str, instances: list[dict] | None = None) -> dict:
        """The entry with this id, or NotConfiguredError naming what is known.

        The id and nothing else. It used to answer to four names — the id, the
        label a person gave it, the model's path, and the file at the end of
        that path — and the first match won.

        That is a collision waiting to be found. Two entries pointing at one
        model with different settings is not a strange thing to want; it is
        exactly what the settings in a request are for. Both would have
        answered to the file's name, one of them would have won silently, and
        the request would have been served by the wrong one.

        So there is one name. It is stable, it survives renaming the label, it
        has no spaces in it, and it fits the single `model` field the request
        shapes give us. The label is for reading; this is for sending.
        """
        key = (wanted or "").strip().lower()
        if instances is None:
            instances = self.operations.configured()
        for instance in instances:
            if instance["id"].strip().lower() == key:
                return instance
        known = sorted(instance["id"] for instance in instances)
        raise NotConfiguredError(
            f"No configured model answers to {wanted!r}. Known: {', '.join(known)}")

    def _answers(self, instance: dict, shape: str) -> bool:
        """Whether this entry's engine answers this kind of request."""
        try:
            engine = self.operations.engines.get(instance["engine"])
        except KeyError:
            return False
        task = Task(instance.get("task", Task.TEXT_GENERATION.value))
        return shape in engine.api_paths(task)

    def _refuse_wrong_shape(self, instance: dict, shape: str,
                            instances: list[dict]) -> None:
        """Say no, and say which entries would have worked.

        A client that sent the wrong shape does not know which of its models
        are on which engine, and has no way to find out from a refusal that
        only says no. Naming them turns a dead end into one edit.
        """
        if self._answers(instance, shape):
            return
        able = sorted(other["id"] for other in instances
                      if self._answers(other, shape))
        raise ShapeNotServedError(
            f"{instance['id']} runs on {instance['engine']}, which does not "
            f"answer {shape}. "
            + (f"Configured models that do: {', '.join(able)}." if able
               else "No configured model answers it."))

    def _engine_name_of(self, shape) -> str:
        if shape is None:
            return ""
        try:
            entry = self.operations.instance(shape.instance_id)
            return self.operations.engines.get(entry["engine"]).display_name
        except Exception:
            return ""
