"""Version labels, release lines and build records for source builds."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Line:
    """One release line upstream publishes: the tags' glob for git, and their exact shape."""

    glob: str
    pattern: re.Pattern[str]


LINES = {
    "stable": Line("v*", re.compile(r"^v\d+\.\d+\.\d+$")),
    "nightly": Line("b*", re.compile(r"^b\d+$")),
}
DEFAULT_LINE = "stable"
META = ".ai-lab-build.json"


@dataclass(frozen=True, slots=True)
class Version:
    """A source version: its tag on one of the lines, and the commit it names."""

    tag: str                      # "b10331" or "v0.2.0", or "" when unknown
    commit: str = ""

    @property
    def line(self) -> str:
        """Which of the two lines this tag belongs to, or "" for neither."""
        for name, shape in LINES.items():
            if shape.pattern.match(self.tag or ""):
                return name
        return ""

    @property
    def number(self) -> tuple[int, ...]:
        """Enough of the tag to order two of them on the *same* line.

        Never used to compare across lines: `b10448` would sort above `v0.2.0`
        and mean nothing by it. Comparing the two lines is git's job — see
        `_is_ahead`.
        """
        return tuple(int(part) for part in re.findall(r"\d+", self.tag or ""))


class BuildEnvironment:
    """One compiled source version, shaped like a package environment."""

    def __init__(self, path: Path, version: Version, active: bool,
                 movable: bool = True) -> None:
        self.path, self.version, self.active = path, version, active
        self.name = path.name
        self.movable = movable
        self.size_bytes = 0

    def json(self) -> dict:
        """The row the settings page shows for this build."""
        return {"name": self.name, "version": self.version.tag,
                "commit": self.version.commit, "active": self.active,
                "size_bytes": self.size_bytes, "path": str(self.path),
                "movable": self.movable}


def marked_version(path: Path) -> Version:
    """The version a build folder records it was compiled from; empty when unknown."""
    try:
        raw = json.loads((path / META).read_text())
        return Version(str(raw.get("tag") or ""), str(raw.get("commit") or ""))
    except (OSError, ValueError, TypeError):
        return Version("")


def mark_version(path: Path, version: Version) -> None:
    """Record in a build folder which version it was compiled from, atomically."""
    temporary = path / f"{META}.tmp"
    temporary.write_text(json.dumps({"tag": version.tag,
                                     "commit": version.commit}) + "\n")
    temporary.replace(path / META)
