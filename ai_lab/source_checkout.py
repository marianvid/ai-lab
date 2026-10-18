"""What a source checkout says about itself: its version, its tags, what is new.

A base class of `SourceBuild` (a mixin): everything here asks git, in the
checkout at `path`, and changes nothing but the checked-out commit.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from .source_versions import DEFAULT_LINE, LINES, Version

# How long one quick git question may take.
GIT_QUESTION_S = 15
# How long listing the tags may take.
TAG_LISTING_S = 30


class GitCheckout:
    """Questions to git about the checkout at `path` (see `SourceBuild`, which owns it)."""

    path: Path

    @property
    def exists(self) -> bool:
        """Whether there is a git checkout at the configured path."""
        return (self.path / ".git").is_dir()

    def checkout_version(self) -> Version:
        """What source is checked out, independently of the active binary."""
        if not self.exists:
            return Version("")
        try:
            tag = self._git("describe", "--tags", "--abbrev=0").strip()
            commit = self._git("rev-parse", "--short", "HEAD").strip()
        except ValueError:
            return Version("")
        return Version(tag, commit)

    def newest_on(self, line: str) -> str:
        """The highest tag on one line, or "" if that line has none here.

        Sorted by git rather than in Python, so that `v0.1.10` comes after
        `v0.1.2` and `b10000` after `b9999` without this file having to know
        how either is spelled.
        """
        shape = LINES.get(line) or LINES[DEFAULT_LINE]
        try:
            output = self._git("for-each-ref", "--sort=-v:refname",
                               "--format=%(refname:short)",
                               f"refs/tags/{shape.glob}", timeout=TAG_LISTING_S)
        except ValueError:
            return ""
        for candidate in output.splitlines():
            if shape.pattern.match(candidate.strip()):
                return candidate.strip()
        return ""

    def _update_available(self, installed: Version, latest: Version | None) -> bool:
        """Is there something to move to?

        Asked of git rather than of the numbers, because the two lines are not
        on one scale: somebody on `b10448` who switches to the stable line is
        moving to `v0.2.0`, which is a smaller number and a newer thing.

        The question git answers is exact: is that tag already part of this
        checkout's history? If it is, there is nothing to do. If it is not,
        there is.

        A checkout with no tags at all — a shallow clone, which is how you
        clone when you only ever meant to build once — cannot answer, and an
        update is not offered rather than offered meaninglessly.
        """
        if latest is None or not latest.tag or not installed.tag:
            return False
        return self._is_ahead(latest.tag, installed.commit or installed.tag or "HEAD")

    def _is_ahead(self, tag: str, against: str = "HEAD") -> bool:
        """Would moving to this tag bring anything this checkout does not have?"""
        try:
            subprocess.run(["git", "merge-base", "--is-ancestor", tag, against],
                           cwd=self.path, capture_output=True, timeout=GIT_QUESTION_S,
                           check=True)
        except subprocess.CalledProcessError:
            return True          # not an ancestor: there is something new
        except (OSError, subprocess.TimeoutExpired):
            return False         # cannot tell, so do not claim there is
        return False

    def _note(self, installed: Version) -> str:
        """A plain sentence about anything that needs fixing by hand."""
        if not self.exists:
            return "No git checkout at this path."
        if not installed.tag:
            return ("This checkout has no version tags, usually because it was "
                    "cloned shallow. Run: git fetch --unshallow --tags")
        return ""

    def _git(self, *arguments: str, timeout: int = GIT_QUESTION_S) -> str:
        """git's output for one command in the checkout; ValueError with its words if it fails."""
        try:
            completed = subprocess.run(
                ["git", *arguments], cwd=self.path, capture_output=True,
                text=True, timeout=timeout)
        except (OSError, subprocess.TimeoutExpired) as error:
            raise ValueError(f"git {arguments[0]} failed: {error}") from None
        if completed.returncode != 0:
            raise ValueError(completed.stderr.strip() or
                             f"git {arguments[0]} failed")
        return completed.stdout
