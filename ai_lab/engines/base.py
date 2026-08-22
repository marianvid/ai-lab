"""The interface every engine must satisfy.

An engine describes itself: the formats it reads, the settings it accepts, the
command that starts it, and how to tell when it is ready. Nothing outside this
package needs to know which engine is in play, which is why adding one touches
neither the configuration schema nor the front end.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..types import Format, ModelSet


@dataclass(frozen=True, slots=True)
class ParamSpec:
    """One tunable setting an engine accepts.

    The same description is used twice: to validate what arrives from the
    browser, and to draw the form that produced it. Keeping them together is
    what stops the two from drifting apart.
    """

    key: str
    label: str
    kind: str                       # "int", "bool", "choice", "float"
    default: object
    minimum: float | None = None
    maximum: float | None = None
    choices: tuple[str, ...] = ()
    help: str = ""
    # "memory": decides how much is reserved on the accelerator, so changing it
    # means reloading the model.
    # "generation": only a default for requests — any client can override it
    # per call, so it is a convenience rather than a constraint.
    group: str = "memory"

    def coerce(self, value: object) -> object:
        """Convert and range-check one incoming value.

        Raises ValueError with a message meant for a person, since it is shown
        in the interface.
        """
        if self.kind == "bool":
            if isinstance(value, bool):
                return value
            raise ValueError(f"{self.label} must be true or false")
        if self.kind in ("int", "float"):
            converter = int if self.kind == "int" else float
            noun = "whole number" if self.kind == "int" else "number"
            try:
                number = converter(value)
            except (TypeError, ValueError):
                raise ValueError(f"{self.label} must be a {noun}") from None
            if self.minimum is not None and number < self.minimum:
                raise ValueError(f"{self.label} must be at least {self.minimum}")
            if self.maximum is not None and number > self.maximum:
                raise ValueError(f"{self.label} must be at most {self.maximum}")
            return number
        if self.kind == "choice":
            text = str(value)
            if text not in self.choices:
                raise ValueError(f"{self.label} must be one of {', '.join(self.choices)}")
            return text
        raise ValueError(f"Unsupported setting kind: {self.kind}")


def validate(specs: tuple[ParamSpec, ...], values: dict) -> dict:
    """Fill in defaults, coerce what was supplied, reject what is unknown.

    Shared by every engine so validation cannot drift between them.
    """
    known = {spec.key: spec for spec in specs}
    unknown = set(values) - set(known)
    if unknown:
        raise ValueError(f"Unknown settings: {', '.join(sorted(unknown))}")
    return {
        key: spec.coerce(values[key]) if key in values else spec.default
        for key, spec in known.items()
    }


# The shapes a request can arrive in. Two are in circulation, and they are two
# ways of writing the same thing rather than two kinds of thing.
#
# OPENAI is what nearly everything speaks, and every engine here answers it.
# ANTHROPIC is the other one; a client written against Anthropic's own library
# sends this, and only some engines understand it. An engine says which it
# answers through `Engine.api_paths`, so the manager never has to know one
# engine from another.
OPENAI_PATHS = ("/v1/chat/completions", "/v1/completions", "/v1/embeddings")
ANTHROPIC_PATHS = ("/v1/messages", "/v1/messages/count_tokens")


@dataclass(frozen=True, slots=True)
class LaunchPlan:
    """The command line for one instance, and how to tell when it is ready."""

    argv: list[str]
    env: dict[str, str]
    health_path: str = "/health"
    # Whether this engine also serves a page a person can talk to, as opposed
    # to an API only. The interface offers a link only when there is one.
    web_ui: bool = False
    # True when this plan deliberately leaves part of the model in system
    # memory. The manager refuses a model bigger than the free memory on the
    # card, because normally that is a crash a few seconds later dressed up as
    # a mystery. When the split is what you asked for, that refusal is wrong,
    # and this is how the engine says so.
    splits_across_cpu: bool = False


class Engine(Protocol):
    id: str
    display_name: str

    def formats(self) -> frozenset[Format]:
        """Weight formats this engine can load."""
        ...

    def params(self) -> tuple[ParamSpec, ...]:
        """The settings this engine accepts."""
        ...

    def plan(self, model: ModelSet, port: int, params: dict) -> LaunchPlan:
        """Build the command line. Pure — nothing is started here."""
        ...

    def ready(self, port: int) -> bool:
        """True once the weights are loaded and the engine will answer.

        Polled repeatedly during a load, so it must be cheap and must never
        raise.
        """
        ...

    def concurrency(self, params: dict) -> int:
        """How many requests this engine serves at once, with these settings.

        Every engine has the idea and none of them spell it the same way:
        llama.cpp counts slots, vLLM counts sequences. The number decides how
        many requests may share the card without one waiting for another, and
        asking the engine is the only way to get it without the manager keeping
        a list of setting names it would have to be told about.
        """
        ...

    def api_paths(self) -> tuple[str, ...]:
        """The request shapes this engine answers.

        Every engine answers `OPENAI_PATHS`. One that also speaks another shape
        adds it here, and the front door then accepts that shape for its
        entries and refuses it for the rest — with a sentence naming which
        entries would have worked, rather than passing the request on to an
        engine that will reject it in its own words.
        """
        ...
