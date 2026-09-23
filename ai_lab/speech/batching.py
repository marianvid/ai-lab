"""Group requests that arrive together so one model pass serves them all.

A speech model without a serving engine generates one request per call. On
a GPU whose limit is reading the weights, advancing several requests in one
pass costs little more than advancing one: measured on the M3 Max with
Higgs, eight lines together were 3.5 times faster than one after another.

This is static batching: requests arriving within a short window form a
group, and the group runs to the end together. A request that arrives while
a group is running waits for the next group. Letting it join the running
group — continuous batching, as vLLM does — is deliberately not done here.
"""
from __future__ import annotations

import threading
from concurrent.futures import Future
from typing import Callable


class BatchQueue:
    """Collect requests for `window_s` after the first, up to `max_batch`.

    `run_batch` receives a list of requests and returns one result per
    request, in order. It runs on a single worker thread, so the model is
    only ever used by one group at a time.
    """

    def __init__(self, run_batch: Callable[[list], list], max_batch: int,
                 window_s: float) -> None:
        if not 1 <= max_batch <= 64 or not 0 <= window_s <= 5:
            raise ValueError("batch size must be 1–64 and the window 0–5 s")
        self.run_batch = run_batch
        self.max_batch = max_batch
        self.window_s = window_s
        self._pending: list[tuple[object, Future]] = []
        self._cond = threading.Condition()
        threading.Thread(target=self._work, daemon=True, name="batch").start()

    def submit(self, request) -> object:
        """Queue one request and wait for its result (or its exception)."""
        future: Future = Future()
        with self._cond:
            self._pending.append((request, future))
            self._cond.notify_all()
        return future.result()

    def _take(self) -> list[tuple[object, Future]]:
        with self._cond:
            while not self._pending:
                self._cond.wait()
            # The first request is here: give its companions a moment to
            # arrive, but never keep a full group waiting.
            if len(self._pending) < self.max_batch:
                self._cond.wait_for(lambda: len(self._pending) >= self.max_batch,
                                    timeout=self.window_s)
            group = self._pending[:self.max_batch]
            del self._pending[:self.max_batch]
            return group

    def _work(self) -> None:
        while True:
            group = self._take()
            try:
                results = self.run_batch([request for request, _ in group])
                if len(results) != len(group):
                    raise RuntimeError("batch returned the wrong number of results")
            except BaseException as error:  # every waiter must hear about it
                for _, future in group:
                    future.set_exception(error)
                continue
            for (_, future), result in zip(group, results):
                if isinstance(result, BaseException):
                    future.set_exception(result)
                else:
                    future.set_result(result)
