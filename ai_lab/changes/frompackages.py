"""What an engine installed as packages would actually replace.

Release notes say what a version does. They do not say what will happen to
*this* machine — and for vLLM that is the more dangerous half. Measured on the
container: moving from the installed nightly to 0.27.1 touches fifteen
packages, and five of them go **backwards**, because the nightly had pulled
newer kernel libraries than the stable release pins.

Nothing here installs anything. `uv` is asked what it *would* do and answers
without touching the environment.

The packages worth calling out by name are the heavy, fragile ones — the CUDA
runtime and the tensor library. When one of those moves, several gigabytes move
with it and the result has to still support this card.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

TIMEOUT_S = 180

# The ones whose movement changes the character of the update rather than
# just its version number.
HEAVY = ("torch", "nvidia-cuda", "nvidia-cudnn", "triton", "flashinfer",
         "transformers", "xformers")

# uv prints "- name==old" then "+ name==new".
LINE = re.compile(r"^\s*([-+])\s*([A-Za-z0-9_.\-]+)==(\S+)\s*$")


class Move:
    """One package, and where it would go."""

    __slots__ = ("name", "was", "becomes")

    def __init__(self, name: str, was: str = "", becomes: str = "") -> None:
        self.name, self.was, self.becomes = name, was, becomes

    @property
    def backwards(self) -> bool:
        """Would this replace what is installed with something older?"""
        return bool(self.was and self.becomes
                    and _number(self.becomes) < _number(self.was))

    @property
    def heavy(self) -> bool:
        return any(self.name.lower().startswith(item) for item in HEAVY)

    def json(self) -> dict:
        return {"name": self.name, "was": self.was, "becomes": self.becomes,
                "backwards": self.backwards, "heavy": self.heavy}


def would_move(venv: str, package: str, uv: str = "uv") -> list[Move]:
    """Ask what updating `package` in this environment would change.

    Raises `ValueError` with something readable. Nothing is installed and
    nothing is written: this is the question, not the act.
    """
    try:
        result = subprocess.run(
            [uv, "pip", "install", "--dry-run", "-U", package],
            capture_output=True, text=True, timeout=TIMEOUT_S,
            env={"VIRTUAL_ENV": venv, "PATH": "/usr/local/bin:/usr/bin:/bin",
                 "HOME": str(Path.home())})
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ValueError(f"could not ask uv what would change: {error}") from None
    if result.returncode != 0:
        first = (result.stderr or result.stdout).strip().splitlines()
        raise ValueError(first[-1] if first else "uv would not answer")

    # uv prints both halves of a replacement as separate lines, so they are
    # collected by name and joined here.
    moves: dict[str, Move] = {}
    for line in (result.stdout + result.stderr).splitlines():
        match = LINE.match(line)
        if not match:
            continue
        sign, name, version = match.groups()
        move = moves.setdefault(name, Move(name))
        if sign == "-":
            move.was = version
        else:
            move.becomes = version
    # Newly added and newly removed packages have only one half, which is
    # true and worth showing: an update that pulls in a new dependency is a
    # bigger change than one that bumps a number.
    return sorted(moves.values(),
                  key=lambda move: (not move.heavy, not move.backwards, move.name))


def installed_version(venv: str, package: str) -> str:
    """Which version of the package is in this environment right now.

    Asked of the environment's own Python rather than worked out from the
    dry-run, because the dry-run says nothing about a package that is already
    up to date — and "up to date at 0.27.1" is exactly what somebody wants to
    read before deciding not to press anything.
    """
    program = ("import importlib.metadata as m;"
               f"print(m.version({package!r}))")
    try:
        result = subprocess.run([str(Path(venv) / "bin" / "python"), "-c", program],
                                capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def venv_of(binary: str) -> str:
    """The environment a launcher belongs to: …/.venv/bin/vllm -> …/.venv"""
    path = Path(binary)
    return str(path.parent.parent) if path.parent.name == "bin" else str(path.parent)


def _number(version: str) -> tuple:
    """Enough of a version to compare two of the same package."""
    parts = re.findall(r"\d+", version.split("+")[0])
    return tuple(int(part) for part in parts[:4])
