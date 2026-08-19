"""Publishing progress to whoever is watching.

One publisher — the runtime — and any number of subscribers, normally one per
open browser tab.

Each subscriber gets its own bounded queue. That bound is the important part:
a browser that stops reading must never be able to slow down or stall a model
load, so when a queue fills, that subscriber's oldest event is dropped rather
than the publisher being made to wait.
"""

from __future__ import annotations

from dataclasses import asdict
from queue import Empty, Full, Queue
from threading import RLock
from typing import Iterator

from .types import ChangeEvent, LogEvent, RuntimeEvent

QUEUE_SIZE = 256


class Subscription:
    """One watcher's view of the stream."""

    def __init__(self, bus: "EventBus", queue: Queue) -> None:
        self._bus = bus
        self._queue = queue

    def events(self, timeout: float = 15.0):
        """Yield events as they arrive; yield None when idle.

        The idle tick lets the caller send a keep-alive so proxies do not close
        a quiet connection.
        """
        while True:
            try:
                yield self._queue.get(timeout=timeout)
            except Empty:
                yield None

    def close(self) -> None:
        self._bus.unsubscribe(self)


class EventBus:
    def __init__(self, queue_size: int = QUEUE_SIZE) -> None:
        self._subscribers: list[tuple[Subscription, Queue]] = []
        self._lock = RLock()
        self._queue_size = queue_size

    def subscribe(self) -> Subscription:
        queue: Queue = Queue(maxsize=self._queue_size)
        subscription = Subscription(self, queue)
        with self._lock:
            self._subscribers.append((subscription, queue))
        return subscription

    def unsubscribe(self, subscription: Subscription) -> None:
        with self._lock:
            self._subscribers = [item for item in self._subscribers
                                 if item[0] is not subscription]

    def publish(self, event) -> None:
        """Never blocks. A full queue loses its oldest event, not the newest."""
        with self._lock:
            subscribers = list(self._subscribers)
        for _, queue in subscribers:
            try:
                queue.put_nowait(event)
            except Full:
                try:
                    queue.get_nowait()
                    queue.put_nowait(event)
                except (Empty, Full):
                    pass

    @property
    def subscriber_count(self) -> int:
        with self._lock:
            return len(self._subscribers)


def to_json(event) -> dict:
    """Tag each event with its kind so the browser can route it.

    Two kinds share the stream: progress through a model load, and lines of
    output from a build.
    """
    payload = asdict(event)
    if isinstance(event, RuntimeEvent):
        payload["kind"] = "runtime"
        payload["phase"] = event.phase.value
    elif isinstance(event, ChangeEvent):
        payload["kind"] = "change"
    else:
        payload["kind"] = "log"
    return payload
