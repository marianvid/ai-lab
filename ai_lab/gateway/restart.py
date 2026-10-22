"""Taking a stalled engine off the card, once, so the next request starts it afresh.

An engine that has requests and processes no token at all for the stall limit
is stuck (`progress.py`). Its requests are cut with "engine stalled"; that frees
nobody while the engine itself stays stuck, so the next requests would stall
too. The owner approved restarting it automatically (22 October 2026).

"Restart" here is the same thing the page's forced stop does, minus the
person: the engine is stopped, every request still waiting for the card is
turned away with the reason, and the gateway forgets what it believed was
loaded. The next request loads the model through the ordinary queue. Loading
it here instead would race that queue: a request arriving meanwhile would
start a second load of the same model.

Many requests to one engine notice the stall at about the same moment; only
the first one restarts it. The work runs on a thread of its own, because the
request that noticed is on its way back to its client.
"""

from __future__ import annotations

import threading
from collections.abc import Callable


class StalledEngines:
    """Restarts a stalled engine once, however many requests report it.

    `restart(instance_id, reason)` does the stopping and the bookkeeping; it is
    the gateway's. `start` runs it — a new thread in service, the caller's own
    thread in a test.
    """

    def __init__(self, restart: Callable[[str, str], None],
                 start: Callable[[Callable[[], None]], None] | None = None) -> None:
        self._restart = restart
        self._start = start or _on_a_thread
        self._lock = threading.Lock()
        self._restarting: set[str] = set()

    def report(self, instance_id: str, reason: str) -> bool:
        """An engine stalled: restart it unless that is already under way. True when this call started it."""
        with self._lock:
            if instance_id in self._restarting:
                return False
            self._restarting.add(instance_id)

        def run() -> None:
            try:
                self._restart(instance_id, reason)
            finally:
                with self._lock:
                    self._restarting.discard(instance_id)
        self._start(run)
        return True


def _on_a_thread(work: Callable[[], None]) -> None:
    threading.Thread(target=work, name="restart-stalled-engine", daemon=True).start()
