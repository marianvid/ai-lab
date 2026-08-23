"""What was on the machine, so it can be put back after a restart.

It used to be that after a reboot the card came up however systemd happened to
have been told months ago — two units enabled by hand in August, and nothing in
the application able to say otherwise. This replaces that: what is loaded is
remembered as it changes, and put back when the manager starts.

**More than one can be loaded now**, so this remembers a list rather than a
single fact, in the order they were loaded. On the way back they are put on in
that order and stop when the next one does not fit — a machine given less
memory than it had, or a reserve raised since, must not be filled past what it
can hold just because it once held it.

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
        """Add this to what is loaded, or move it to the end if it is there.

        The order is the order they were loaded, which is the order to put them
        back in: the oldest was there first and is likeliest to be the one that
        matters.
        """
        kept = [item for item in self.all() if item["instance_id"] != instance_id]
        kept.append({"instance_id": instance_id, "settings": settings or {}})
        self._write(kept)

    def forget(self, instance_id: str | None = None) -> None:
        """Record that this is no longer loaded, or that nothing is.

        `instance_id` names one to drop and leaves the rest. Without it the
        whole lot goes, which is what emptying the machine means.
        """
        if instance_id is None:
            self._write([])
            return
        self._write([item for item in self.all()
                     if item["instance_id"] != instance_id])

    def read(self) -> dict | None:
        """The last one loaded, or None. Kept for callers that want just one."""
        found = self.all()
        return found[-1] if found else None

    def all(self) -> list[dict]:
        """Everything that was loaded, oldest first.

        A file that cannot be parsed is treated as no memory at all. The worst
        that costs is a model not coming back on its own; refusing to start
        because of it would be far worse. A file written when this held a
        single fact still reads, as a list of one.
        """
        try:
            stored = json.loads(self.path.read_text())
        except (OSError, ValueError):
            return []
        rows = stored if isinstance(stored, list) else [stored]
        found = []
        for item in rows:
            if not isinstance(item, dict) or not item.get("instance_id"):
                continue
            settings = item.get("settings")
            found.append({"instance_id": str(item["instance_id"]),
                          "settings": settings if isinstance(settings, dict) else {}})
        return found

    # -- writing -----------------------------------------------------------

    def _write(self, value: list) -> None:
        """Write, or fail quietly.

        Losing the memory means a model does not come back by itself. Raising
        here would mean a load or an unload reporting failure for something
        that worked, which is worse.
        """
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix(".tmp")
            temporary.write_text(json.dumps(value or [], indent=2))
            temporary.replace(self.path)
        except OSError:
            pass
