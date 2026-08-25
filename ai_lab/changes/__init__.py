"""What an update would bring, before anybody presses anything.

An update should be a decision, not a hope. This package answers one question —
*what would change if I updated this engine?* — and the answer is assembled
from whatever sources that engine has:

    fromgit.py        commits waiting in a checkout      (llama.cpp)
    fromgithub.py     release notes written upstream     (vLLM)
    frompackages.py   which packages would be replaced   (vLLM)
    sifting.py        which of them matter on this machine

The two engines are genuinely different and are not forced to look alike.
llama.cpp is a checkout, so its changes are commits: many, unwritten for a
reader, and worth sorting. vLLM arrives as packages, so its changes are notes
somebody wrote plus the list of files that would be replaced.

Nothing here decides anything or changes anything. It reads, and hands back
what it found.
"""

from __future__ import annotations

from ..types import Change, Changes, Interests
from .fromgit import waiting
from .fromgithub import between
from .frompackages import installed_version, venv_of, would_move
from .sifting import area_of, counted, sift

__all__ = ["Change", "Changes", "Interests", "area_of", "between", "counted",
           "installed_version", "sift", "venv_of", "waiting", "would_move",
           "Reader"]


class Reader:
    """Answers "what would change?" for one engine.

    Built from that engine's `source` section, which says what it has: a `path`
    means a checkout to read commits from, a `releases` name means a repository
    on GitHub with notes, and a `package` name means an installed package whose
    replacement can be asked about.

    An engine may have any combination. Missing sources are not errors — an
    engine with none simply reports that nothing can be read about it, which is
    honest and still lets the page draw.
    """

    def __init__(self, engine_id: str, source: dict | None, binary: str = "",
                 uv: str = "uv") -> None:
        self.engine_id = engine_id
        source = source or {}
        self.checkout = source.get("path") or ""
        self.releases = source.get("releases") or ""
        self.package = source.get("package") or ""
        self.install = source.get("install") or self.package
        self.binary = binary
        self.uv = uv

    def read(self, installed: str, latest: str, interests: Interests) -> Changes:
        """Everything that can be found out about this update.

        Each source is asked separately and a failure in one does not lose the
        others: no network still leaves the commits, and an unreadable checkout
        still leaves the notes. What could not be read is said plainly rather
        than left as an empty section somebody would read as "nothing changes".
        """
        troubles: list[str] = []
        yours: tuple[Change, ...] = ()
        others: tuple[Change, ...] = ()
        notes = ""

        if self.checkout:
            try:
                found, cut = waiting(self.checkout)
                yours, others = sift(found, interests)
                if cut:
                    troubles.append(
                        f"{cut} older changes beyond the {len(found)} shown "
                        "were not read.")
            except ValueError as error:
                troubles.append(f"Could not read the checkout: {error}")

        if self.releases:
            found, trouble = between(self.releases, installed, latest)
            notes = found
            if trouble:
                troubles.append(trouble)

        return Changes(installed=installed, latest=latest, yours=yours,
                       others=others, notes=notes,
                       unreadable=" ".join(troubles))

    def versions(self, moves: list) -> tuple[str, str]:
        """What is installed and what it would become, for a package engine.

        Read from the environment itself, then from the dry-run: the first
        always answers, the second only when something would actually move. An
        engine already at the newest version reports the same string twice,
        which is how the page says "nothing to do".
        """
        if not self.package or not self.binary:
            return "", ""
        installed = installed_version(venv_of(self.binary), self.package)
        canonical = lambda name: name.lower().replace("_", "-").replace(".", "-")
        moving = next((move for move in moves
                       if canonical(move.name) == canonical(self.package)), None)
        latest = moving.becomes if moving and moving.becomes else installed
        return installed or (moving.was if moving else ""), latest

    def moves(self) -> tuple[list, str]:
        """Which packages would be replaced, for an engine installed as one.

        Empty for a checkout: nothing is replaced there, it is rebuilt.
        """
        if not self.package or not self.binary:
            return [], ""
        try:
            return would_move(venv_of(self.binary), self.install, self.uv), ""
        except ValueError as error:
            return [], f"Could not work out which packages would change: {error}"
