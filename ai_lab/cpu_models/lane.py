"""The embeddings lane of the gateway, with a queue of its own.

`CpuLane` is what the front door asks when a request names one of these
models. It never takes a card lease, so it neither waits behind a model loading
on the card nor holds one up. Each model has its own queue: at most `parallel`
requests reach its server at once; the rest wait here.

Where a request is answered — on the card when there is room, else on the
processor — is `Placement`'s decision (`placement.py`). The answer says which,
in `ai_lab.device`.

A model is started by its first request (a second or two) and stopped when
nothing has used it for `idle_stop_s`. It can also be started and stopped by
hand (`POST /api/cpu-models/{name}/start|stop`).

The lane is also the card's neighbour (`card_sharing.CardNeighbours`): the
card's scheduler calls `make_way` before loading a card model and `settled`
after, and counts `reclaimable_mb` as memory it may have.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ..network import LOOPBACK
from .placement import CardRoom, Placement
from .server import CARD, CPU, CpuServer
from .settings import CpuModel, CpuModelsConfig

EMBEDDINGS_PATH = "/v1/embeddings"
# A batch of long texts on a busy processor can take a while; the queue wait
# is not counted here, only the call to the server.
CALL_LIMIT_S = 600.0
REAP_EVERY_S = 30.0

Poster = Callable[[int, dict], dict]
ServerFactory = Callable[[str, CpuModel, Path, str], CpuServer]


def _server(binary: str, model: CpuModel, log_dir: Path, device: str) -> CpuServer:
    """The real server on that device, ready when its `/health` answers."""
    return CpuServer(binary, model, log_dir, device=device)


def _no_card() -> float | None:
    """No card to borrow: everything answers on the processor."""
    return None


def _post(port: int, payload: dict) -> dict:
    """POST to the model's own server and return its JSON answer."""
    request = urllib.request.Request(
        f"http://{LOOPBACK}:{port}{EMBEDDINGS_PATH}", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=CALL_LIMIT_S) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        # The server refused the request itself (a text too long, say): the
        # client's fault, with the server's own words.
        raise ValueError(error.read().decode(errors="replace")) from None


@dataclass(frozen=True)
class LaneTools:
    """What the lane talks to: servers, the card's free memory, and time.

    A test swaps in stand-ins.
    """

    poster: Poster = _post
    server_factory: ServerFactory = _server
    card_free_mb: CardRoom = _no_card
    clock: Callable[[], float] = time.monotonic
    sleep: Callable[[float], None] = time.sleep


@dataclass
class _Queue:
    """One model's places and counters."""

    places: threading.BoundedSemaphore
    in_flight: int = 0
    waiting: int = 0
    requests: int = 0
    failures: int = 0


class CpuLane:
    """Every embeddings model, each behind its own queue."""

    def __init__(self, config: CpuModelsConfig, log_dir: Path,
                 tools: LaneTools | None = None) -> None:
        tools = tools or LaneTools()
        self.config = config
        self._post = tools.poster
        self._clock = tools.clock
        self._count = threading.Lock()
        self.placements = {name: self._placement(model, log_dir, tools)
                           for name, model in config.models.items()}
        self._queues = {name: _Queue(threading.BoundedSemaphore(model.parallel))
                        for name, model in config.models.items()}

    def _placement(self, model: CpuModel, log_dir: Path, tools: LaneTools) -> Placement:
        make = tools.server_factory
        cpu = make(self.config.binary, model, log_dir, CPU)
        card = make(self.config.binary, model, log_dir, CARD) if model.card else None
        return Placement(cpu, card, tools.card_free_mb, tools.clock, tools.sleep)

    @classmethod
    def from_config(cls, raw: dict | None, log_dir: Path,
                    card_free_mb: CardRoom = _no_card) -> CpuLane:
        """Built from the `cpu_models` section, logging into `log_dir`."""
        return cls(CpuModelsConfig.from_mapping(raw), log_dir,
                   LaneTools(card_free_mb=card_free_mb))

    def serves(self, name: str) -> bool:
        """Whether this name is one of these models."""
        return name in self.placements

    def embed(self, path: str, payload: dict) -> dict:
        """Answer one embeddings request through the model's own server."""
        if path != EMBEDDINGS_PATH:
            raise ValueError(f"processor models answer only {EMBEDDINGS_PATH}")
        name = payload["model"]
        placement, queue = self.placements[name], self._queues[name]
        self._change(queue, waiting=1)
        with queue.places:
            self._change(queue, waiting=-1, in_flight=1, requests=1)
            try:
                server = placement.take()
                try:
                    answer = self._post(server.port, payload)
                finally:
                    placement.give_back(server)
            except BaseException:
                self._change(queue, failures=1)
                raise
            finally:
                self._change(queue, in_flight=-1)
        answer["model"] = name
        answer["ai_lab"] = {"device": server.device}
        return answer

    # -- the card's neighbour -------------------------------------------------

    def make_way(self) -> None:
        """A card model is about to load: every model leaves the card."""
        for placement in self.placements.values():
            placement.make_way()

    def settled(self) -> None:
        """That load is over: models still in use come back if there is room."""
        for placement in self.placements.values():
            placement.settled()

    def reclaimable_mb(self) -> float:
        """Card memory these models hold and give back to any card model."""
        return sum(placement.reclaimable_mb() for placement in self.placements.values())

    # -- by hand, and when idle ----------------------------------------------

    def start(self, name: str) -> dict:
        """Start a model by hand; it answers at once afterwards."""
        self._known(name).start()
        return self.stats()[name]

    def stop(self, name: str) -> dict:
        """Stop a model by hand, unless a request is using it."""
        placement = self._known(name)
        if self._busy(name):
            raise ValueError(f"{name} is answering requests; stop it when they are done")
        placement.stop()
        return self.stats()[name]

    def stop_all(self) -> None:
        """Stop every server, when the manager quits."""
        for placement in self.placements.values():
            placement.stop()

    def reap_idle(self) -> list[str]:
        """Stop every server nothing has used for its idle limit; `name/device` stopped."""
        return [f"{name}/{device}" for name, placement in self.placements.items()
                for device in placement.stop_if_idle(busy=self._busy(name))]

    def start_reaper(self) -> threading.Thread:
        """Check for idle models in the background, every `REAP_EVERY_S`."""
        def loop() -> None:
            while True:
                time.sleep(REAP_EVERY_S)
                self.reap_idle()
        thread = threading.Thread(target=loop, daemon=True, name="cpu-models-reaper")
        thread.start()
        return thread

    # -- what it says about itself --------------------------------------------

    def catalogue(self) -> list[dict]:
        """One row per model, in the shape `GET /v1/models` lists."""
        return [{"id": name, "object": "model", "owned_by": "llama.cpp",
                 "ai_lab": {"lane": "cpu", "loaded": placement.running,
                            "ready": placement.running, "device": placement.device(),
                            "shapes": [EMBEDDINGS_PATH], "task": "embeddings"}}
                for name, placement in self.placements.items()]

    def stats(self) -> dict:
        """Per model: where it answers, its queue, and its counts since start."""
        return {name: self._stats(name, placement)
                for name, placement in self.placements.items()}

    def _stats(self, name: str, placement: Placement) -> dict:
        queue, model = self._queues[name], placement.model
        last = placement.last_used or placement.cpu.last_used
        return {"running": placement.running, "device": placement.device(),
                "port": model.port, "card": bool(model.card),
                "card_port": model.card.port if model.card else None,
                "card_in_flight": placement.on_card(), "places": model.parallel,
                "in_flight": queue.in_flight, "waiting": queue.waiting,
                "requests": queue.requests, "failures": queue.failures,
                "idle_s": round(self._clock() - last, 1) if last else None,
                "idle_stop_s": model.idle_stop_s}

    def _known(self, name: str) -> Placement:
        if name not in self.placements:
            raise KeyError(f"Unknown processor model: {name}")
        return self.placements[name]

    def _busy(self, name: str) -> bool:
        queue = self._queues[name]
        return bool(queue.in_flight or queue.waiting)

    def _change(self, queue: _Queue, **deltas: int) -> None:
        with self._count:
            for key, delta in deltas.items():
                setattr(queue, key, getattr(queue, key) + delta)
