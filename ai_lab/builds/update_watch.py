"""Asking upstream for new versions on a timer, in the background.

So the settings screen is already right when it is opened, rather than only
after someone presses a button. Shared by the source builds (`builds.py`) and
the package installs (`installs.py`). A failure is ignored: being offline, or
upstream being unreachable, is not worth a message on a page nobody is
looking at.
"""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Callable, Iterable
from typing import Protocol

from ..config.policy import UPDATE_CHECK_EVERY_S


class Checkable(Protocol):
    """Something that can ask upstream whether a newer version exists."""

    def check(self) -> object:
        """Ask now; the answer is kept by the item itself."""
        ...


class UpdateWatcher:
    """One background thread checking every item, then waiting `interval_s`."""

    def __init__(self, items: Callable[[], Iterable[Checkable]], name: str,
                 every_s: float = UPDATE_CHECK_EVERY_S, first_s: float = 0.0) -> None:
        self._items = items
        self._name = name
        # The timer's own defaults (`limits.builds`): every `every_s`, the first after `first_s`.
        self._every_s, self._first_s = every_s, first_s
        self._stop = threading.Event()
        self.thread: threading.Thread | None = None

    def start(self, interval_s: float | None = None,
              startup_delay_s: float | None = None) -> None:
        """Start checking, once: a second call, or nothing to check, does nothing.

        The first check waits `startup_delay_s`, so starting up is not held
        back by a network call nobody asked for yet. Either left out: the
        watcher's own default.
        """
        if self.thread is not None or not list(self._items()):
            return
        interval_s = self._every_s if interval_s is None else interval_s
        startup_delay_s = self._first_s if startup_delay_s is None else startup_delay_s
        self.thread = threading.Thread(target=self._loop, args=(interval_s, startup_delay_s),
                                       daemon=True, name=self._name)
        self.thread.start()

    def stop(self) -> None:
        """Stop after the check in progress, if any."""
        self._stop.set()

    def _loop(self, interval_s: float, startup_delay_s: float) -> None:
        while not self._stop.wait(startup_delay_s):
            for item in self._items():
                if self._stop.is_set():
                    return
                with contextlib.suppress(Exception):
                    item.check()
            if self._stop.wait(interval_s):
                return
