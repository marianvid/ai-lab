"""Request lease and immutable loaded-model shape for the gateway."""

from __future__ import annotations

import time
from dataclasses import dataclass

@dataclass(slots=True)
class Lease:
    """A place on the card, held for the length of one request.

    Taken before the request is forwarded and given back after the last byte of
    the answer, including a streamed answer that takes a minute.

    Each lease knows whether it has been given back, because several are held
    at once now. A request that fails while being forwarded hands its place
    back twice — once from the code that noticed, once from the reader's
    cleanup — and without this the second would be taking a place from somebody
    else.
    """

    gateway: "Gateway"
    instance_id: str
    port: int
    # What the engine calls its own model. llama.cpp and vLLM are both started
    # with an explicit name and both refuse a request naming anything else, so
    # the name the client used has to be translated before forwarding. It is
    # carried here because it was known when the lease was made: asking for it
    # afterwards meant reading every instance's state a second time, which is
    # the expensive question, for an answer that is pure configuration.
    model_name: str = ""
    # When the place was taken, so giving it back can say how long it was held.
    started: float = 0.0
    _given_back: bool = False

    def release(self) -> None:
        if self._given_back:
            return
        self._given_back = True
        self.gateway.finished(time.perf_counter() - self.started
                              if self.started else 0.0)

    def __enter__(self) -> "Lease":
        return self

    def __exit__(self, *_exception) -> None:
        self.release()


@dataclass(frozen=True, slots=True)
class Shape:
    """A model, and the settings it has to be started with.

    Two requests for the same entry wanting different context sizes are not
    requests for the same thing — one of them needs a reload — so the settings
    are part of what is being asked for, not a note attached to it. Frozen and
    hashable, so equality is the whole test.
    """

    instance_id: str
    settings: tuple = ()                # (key, value) pairs, sorted

    @classmethod
    def of(cls, instance_id: str, settings: dict | None) -> "Shape":
        return cls(instance_id, tuple(sorted((settings or {}).items())))

    def as_dict(self) -> dict:
        return dict(self.settings)
