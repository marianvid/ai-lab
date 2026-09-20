"""Version labels, environment records and disk helpers for source builds."""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

LINES = {
    "stable": {"glob": "v*", "pattern": re.compile(r"^v\d+\.\d+\.\d+$")},
    "nightly": {"glob": "b*", "pattern": re.compile(r"^b\d+$")},
}
DEFAULT_LINE = "stable"
META = ".ai-lab-build.json"

@dataclass(frozen=True, slots=True)
class Version:
    tag: str                      # "b10331" or "v0.2.0", or "" when unknown
    commit: str = ""

    @property
    def line(self) -> str:
        """Which of the two lines this tag belongs to, or "" for neither."""
        for name, shape in LINES.items():
            if shape["pattern"].match(self.tag or ""):
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
        return {"name": self.name, "version": self.version.tag,
                "commit": self.version.commit, "active": self.active,
                "size_bytes": self.size_bytes, "path": str(self.path),
                "movable": self.movable}


def _marked(path: Path) -> Version:
    try:
        raw = json.loads((path / META).read_text())
        return Version(str(raw.get("tag") or ""), str(raw.get("commit") or ""))
    except (OSError, ValueError, TypeError):
        return Version("")


def _mark(path: Path, version: Version) -> None:
    temporary = path / f"{META}.tmp"
    temporary.write_text(json.dumps({"tag": version.tag,
                                     "commit": version.commit}) + "\n")
    temporary.replace(path / META)


def _size(path: Path) -> int:
    total = 0
    try:
        for item in path.rglob("*"):
            try:
                if item.is_file() and not item.is_symlink():
                    total += item.stat().st_size
            except OSError:
                pass
    except OSError:
        pass
    return total


def _free(path: Path | None) -> int:
    try:
        return shutil.disk_usage(path).free if path else 0
    except OSError:
        return 0
