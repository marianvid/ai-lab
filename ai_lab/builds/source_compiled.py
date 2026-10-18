"""The compiled versions of one source checkout, and which of them runs.

A base class of `SourceBuild` (a mixin). With a versioned build root, every
compiled version lives in its own `build-<tag>` folder beside the others and
a `current` link names the one the engine runs; switching is moving the link,
so going back to an older version is instant and deletes nothing.
"""

from __future__ import annotations

import contextlib
import re
import shutil
from collections.abc import Callable
from pathlib import Path

from ..disk import tree_size
from .source_versions import BuildEnvironment, Version, marked_version

CURRENT = "current"
BUILD_NAME = re.compile(r"^build-(b\d+|v\d+\.\d+\.\d+)$")
# How long a checkout to another version may take.
CHECKOUT_S = 120


class CompiledVersions:
    """Compiled builds under `build_root` (see `SourceBuild`, which owns the fields)."""

    build_root: Path | None
    legacy_build: Path | None
    # From `GitCheckout`, the other base class of `SourceBuild`.
    _git: Callable[..., str]
    checkout_version: Callable[[], Version]
    status: Callable[..., dict]
    _announce: Callable[[], None]

    @property
    def link(self) -> Path | None:
        """The `current` link naming the build in use; None without a build root."""
        return self.build_root / CURRENT if self.build_root else None

    def active(self) -> Path | None:
        """The build folder the engine runs from now, or None."""
        if self.link is None:
            return None
        try:
            return self.link.resolve(strict=True)
        except OSError:
            return None

    def environments(self, with_sizes: bool = True) -> list[BuildEnvironment]:
        """The active source build and every compiled rollback beside it."""
        if not self.build_root:
            return []
        active = self.active()
        found = []
        seen = set()
        for path, movable in self._candidates(self.build_root):
            try:
                resolved = path.resolve()
            except OSError:
                continue
            if resolved in seen:
                continue
            seen.add(resolved)
            version = marked_version(path)
            if not version.tag and path == self.legacy_build and active == resolved:
                version = self.checkout_version()
            environment = BuildEnvironment(path, version, active == resolved, movable)
            if with_sizes:
                environment.size_bytes = tree_size(path)
            found.append(environment)
        return found

    def _candidates(self, build_root: Path) -> list[tuple[Path, bool]]:
        """The legacy build (which cannot move) and every `build-<tag>` folder."""
        candidates: list[tuple[Path, bool]] = []
        if self.legacy_build and self.legacy_build.is_dir():
            candidates.append((self.legacy_build, False))
        with contextlib.suppress(OSError):
            candidates.extend((entry, True) for entry in sorted(build_root.iterdir())
                              if entry.is_dir() and BUILD_NAME.match(entry.name))
        return candidates

    def point_at(self, build: Path) -> None:
        """Make `build` the one the engine runs, by replacing the `current` link at once."""
        if self.link is None or self.build_root is None:
            raise ValueError("This source build has no versioned build root")
        if not (build / "bin" / "llama-server").is_file():
            raise ValueError(f"{build.name} has no bin/llama-server")
        temporary = self.build_root / f".{CURRENT}-new"
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()
        temporary.symlink_to(build, target_is_directory=True)
        temporary.replace(self.link)

    def activate(self, name: str) -> dict:
        """Run the compiled build called `name`, with the checkout moved to its source."""
        environment = self._one(name)
        reference = environment.version.commit or environment.version.tag
        if not reference:
            raise ValueError(f"{name} does not record which source it was built from")
        # Keep the source and the selected binary describing the same version.
        # This is a quick checkout, not a rebuild, and makes the next update
        # comparison start from what actually runs.
        previous = self.checkout_version()
        self._git("checkout", "--force", reference, timeout=CHECKOUT_S)
        try:
            self.point_at(environment.path)
        except Exception:
            self._back_to(previous)
            raise
        self._announce()
        return self.status()

    def remove(self, name: str) -> dict:
        """Delete a compiled build that is not the one in use."""
        environment = self._one(name)
        if environment.active:
            raise ValueError(f"{name} is the build in use. Activate another one first.")
        shutil.rmtree(environment.path)
        self._announce()
        return self.status()

    def _back_to(self, previous: Version) -> None:
        """Check the source out at `previous` again, if it is known; best effort."""
        if previous.tag:
            with contextlib.suppress(ValueError):
                self._git("checkout", "--force", previous.commit or previous.tag,
                          timeout=CHECKOUT_S)

    def _one(self, name: str) -> BuildEnvironment:
        for environment in self.environments(with_sizes=False):
            if environment.name == name:
                return environment
        raise KeyError(f"No compiled build called {name}")
