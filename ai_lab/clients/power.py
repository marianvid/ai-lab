"""The On/Off switch (A6).

Off: the calls running now are ended at once, those waiting are turned away,
and every new call that would use a model is refused with 503 `ai_lab_off`.
Reading the page, the settings and the switch itself stays open. The switch
survives a restart: an AI-Lab switched Off comes back Off.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from pathlib import Path

from .registry import write_private

# Calls that would use a model: refused while Off.
MODEL_PREFIXES = ("/v1/", "/tokenize", "/api/media-jobs", "/api/image-jobs")
LOAD_SUFFIX = "/load"
INSTANCES = "/api/instances/"


def uses_a_model(method: str, path: str) -> bool:
    """Whether this call would put a model to work (refused while Off)."""
    if method.upper() == "GET":
        return False                            # reading (the model list too) stays open
    if path.startswith(MODEL_PREFIXES):
        return True
    return path.startswith(INSTANCES) and path.endswith(LOAD_SUFFIX)


class Power:
    """On or Off, kept in one small private file, with what to do on Off."""

    def __init__(self, path: Path, clock: Callable[[], float] = time.time) -> None:
        self.path = Path(path)
        self.clock = clock
        self._lock = threading.Lock()
        self._state = json.loads(self.path.read_text()) if self.path.is_file() else {"on": True}
        self._when_off: list[Callable[[], object]] = []

    @property
    def on(self) -> bool:
        """Whether AI-Lab serves calls."""
        return bool(self._state.get("on", True))

    def view(self) -> dict:
        """On or Off, and since when."""
        return {"on": self.on, "since": self._state.get("since")}

    def when_off(self, action: Callable[[], object]) -> None:
        """Something to do each time AI-Lab is switched Off."""
        self._when_off.append(action)

    def switch(self, on: bool) -> dict:
        """Switch On or Off; switching Off runs the Off actions."""
        with self._lock:
            changed = bool(on) != self.on
            if changed:
                self._state = {"on": bool(on), "since": self.clock()}
                write_private(self.path, self._state)
        if changed and not on:
            for action in self._when_off:
                action()
        return self.view()
