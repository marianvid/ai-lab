"""The processor lane of the gateway: embeddings, with a queue of their own.

`CpuLane` is what the front door asks when a request names a processor model.
It never takes a card lease, so it neither waits behind a model loading on
the card nor holds one up. Each model has its own queue: at most `parallel`
requests reach its server at once; the rest wait here.

A model is started by its first request (a second or two) and stopped when
nothing has used it for `idle_stop_s`. It can also be started and stopped by
hand (`POST /api/cpu-models/{name}/start|stop`).
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

from ..engines.probe import http_ok
from ..network import LOOPBACK
from .server import CpuServer
from .settings import CpuModel, CpuModelsConfig

EMBEDDINGS_PATH = "/v1/embeddings"
# A batch of long texts on a busy processor can take a while; the queue wait
# is not counted here, only the call to the server.
CALL_LIMIT_S = 600.0
REAP_EVERY_S = 30.0

Poster = Callable[[int, dict], dict]
ServerFactory = Callable[[str, CpuModel, Path], CpuServer]


def _server(binary: str, model: CpuModel, log_dir: Path) -> CpuServer:
    """The real server, ready when its `/health` answers."""
    return CpuServer(binary, model, log_dir, http_ok)


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


@dataclass
class _Queue:
    """One model's places and counters."""

    places: threading.BoundedSemaphore
    in_flight: int = 0
    waiting: int = 0
    requests: int = 0
    failures: int = 0


class CpuLane:
    """Every processor model, each behind its own queue."""

    def __init__(self, config: CpuModelsConfig, log_dir: Path, poster: Poster = _post,
                 server_factory: ServerFactory = _server,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.config = config
        self._post = poster
        self._clock = clock
        self._count = threading.Lock()
        self.servers = {name: server_factory(config.binary, model, log_dir)
                        for name, model in config.models.items()}
        self._queues = {name: _Queue(threading.BoundedSemaphore(model.parallel))
                        for name, model in config.models.items()}

    @classmethod
    def from_config(cls, raw: dict | None, log_dir: Path) -> CpuLane:
        """Built from the `cpu_models` section, logging into `log_dir`."""
        return cls(CpuModelsConfig.from_mapping(raw), log_dir)

    def serves(self, name: str) -> bool:
        """Whether this name is a processor model."""
        return name in self.servers

    def embed(self, path: str, payload: dict) -> dict:
        """Answer one embeddings request through the model's own server."""
        if path != EMBEDDINGS_PATH:
            raise ValueError(f"processor models answer only {EMBEDDINGS_PATH}")
        name = payload["model"]
        server, queue = self.servers[name], self._queues[name]
        self._change(queue, waiting=1)
        with queue.places:
            self._change(queue, waiting=-1, in_flight=1, requests=1)
            try:
                server.ensure_started()
                answer = self._post(server.model.port, payload)
            except BaseException:
                self._change(queue, failures=1)
                raise
            finally:
                server.touch()
                self._change(queue, in_flight=-1)
        answer["model"] = name
        return answer

    def start(self, name: str) -> dict:
        """Start a model by hand; it answers at once afterwards."""
        self._known(name).ensure_started()
        return self.stats()[name]

    def stop(self, name: str) -> dict:
        """Stop a model by hand, unless a request is using it."""
        server, queue = self._known(name), self._queues[name]
        if queue.in_flight or queue.waiting:
            raise ValueError(f"{name} is answering requests; stop it when they are done")
        server.stop()
        return self.stats()[name]

    def stop_all(self) -> None:
        """Stop every server, when the manager quits."""
        for server in self.servers.values():
            server.stop()

    def reap_idle(self) -> list[str]:
        """Stop every model nothing has used for its idle limit; names stopped."""
        return [name for name, server in self.servers.items()
                if server.stop_if_idle(busy=self._busy(name))]

    def start_reaper(self) -> threading.Thread:
        """Check for idle models in the background, every `REAP_EVERY_S`."""
        def loop() -> None:
            while True:
                time.sleep(REAP_EVERY_S)
                self.reap_idle()
        thread = threading.Thread(target=loop, daemon=True, name="cpu-models-reaper")
        thread.start()
        return thread

    def catalogue(self) -> list[dict]:
        """One row per model, in the shape `GET /v1/models` lists."""
        return [{"id": name, "object": "model", "owned_by": "llama.cpp",
                 "ai_lab": {"lane": "cpu", "loaded": server.running,
                            "ready": server.running, "shapes": [EMBEDDINGS_PATH],
                            "task": "embeddings"}}
                for name, server in self.servers.items()]

    def stats(self) -> dict:
        """Per model: running or not, its queue, and its counts since start."""
        now = self._clock()
        return {name: {"running": server.running, "port": server.model.port,
                       "places": server.model.parallel,
                       "in_flight": self._queues[name].in_flight,
                       "waiting": self._queues[name].waiting,
                       "requests": self._queues[name].requests,
                       "failures": self._queues[name].failures,
                       "idle_s": round(now - server.last_used, 1) if server.last_used else None,
                       "idle_stop_s": server.model.idle_stop_s}
                for name, server in self.servers.items()}

    def _known(self, name: str) -> CpuServer:
        if name not in self.servers:
            raise KeyError(f"Unknown processor model: {name}")
        return self.servers[name]

    def _busy(self, name: str) -> bool:
        queue = self._queues[name]
        return bool(queue.in_flight or queue.waiting)

    def _change(self, queue: _Queue, **deltas: int) -> None:
        with self._count:
            for key, delta in deltas.items():
                setattr(queue, key, getattr(queue, key) + delta)
