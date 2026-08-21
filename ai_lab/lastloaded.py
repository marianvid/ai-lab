"""What was on the card, so it can be put back after a restart.

The rule is one model at a time, and it used to be that after a reboot the
card came up however systemd happened to have been told months ago — two units
enabled by hand in August, and nothing in the application able to say
otherwise. This replaces that: what is on the card is remembered as it
changes, and put back when the manager starts.

**Deliberately unloading is a state too.** Somebody who empties the card and
then reboots does not want the model back. So "nothing" is written down as
firmly as a model is.

**Nothing is restored while something is already running.** On Linux systemd
owns the engines and they survive a manager restart, which is the point of
using it — so a manager coming back finds its model still answering and leaves
it alone. Only a machine that actually rebooted has anything to put back.

**The settings are remembered too.** A request can ask for a model started with
a bigger context than the entry is configured for, and it would be a poor
restore that brought back the same model set up differently.
"""

from __future__ import annotations

import json
from pathlib import Path

FILE_NAME = "last-loaded.json"


class LastLoaded:
    """A single fact on disk: which model was on the card, and how."""

    def __init__(self, directory: Path) -> None:
        self.path = Path(directory) / FILE_NAME

    def remember(self, instance_id: str, settings: dict | None = None) -> None:
        self._write({"instance_id": instance_id, "settings": settings or {}})

    def forget(self, instance_id: str | None = None) -> None:
        """Record that the card is empty.

        `instance_id` guards against forgetting the wrong thing. A stray engine
        is unloaded from beside the model that stays, and that must not be read
        as the card having been emptied.
        """
        if instance_id is not None:
            current = self.read()
            if current and current["instance_id"] != instance_id:
                return
        self._write(None)

    def read(self) -> dict | None:
        """What was last on the card, or None for nothing — and for unreadable.

        A file that cannot be parsed is treated as no memory at all. The worst
        that costs is a model not coming back on its own; refusing to start
        because of it would be far worse.
        """
        try:
            stored = json.loads(self.path.read_text())
        except (OSError, ValueError):
            return None
        if not isinstance(stored, dict) or not stored.get("instance_id"):
            return None
        settings = stored.get("settings")
        return {"instance_id": str(stored["instance_id"]),
                "settings": settings if isinstance(settings, dict) else {}}

    # -- writing -----------------------------------------------------------

    def _write(self, value: dict | None) -> None:
        """Write, or fail quietly.

        Losing the memory means a model does not come back by itself. Raising
        here would mean a load or an unload reporting failure for something
        that worked, which is worse.
        """
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix(".tmp")
            temporary.write_text(json.dumps(value or {}, indent=2))
            temporary.replace(self.path)
        except OSError:
            pass
