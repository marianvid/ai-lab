"""Deleting and moving model files with durable, recoverable move jobs."""

from __future__ import annotations

import json
import os
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ..catalog.core import Catalog
from ..config.store import ConfigStore, ModelRoot, Repository
from ..disk import sha256_of
from ..hosts.base import Host
from .move_files import (
    MovePlan,
    prune_empty,
    reject_unfit_destination,
    reject_unremovable_sources,
)
from .move_jobs import UNFINISHED_MOVE_STATUSES, MoveCancelledError, MoveJobStore


# Makes the move-job store from the host and the change callback; the store
# class itself by default, a fake of it in a test.
MoveJobsFactory = Callable[[Host, Callable[[str], None]], MoveJobStore]


def publish_staged(staging: Path, target_path: Path,
                   relatives: list[Path], destinations: list[Path]) -> None:
    """Make the staged copy visible, as close to one atomic act as the
    filesystem allows.

    When every file sits under one shared top-level folder — the
    ordinary case, a model in its own directory — the whole folder is
    renamed into place in a single `os.replace`, so a crash mid-publish
    either has not happened yet or has already finished; there is no
    state where the destination holds half a model under its real name.

    When files sit loose (no shared folder — the GGUF-in-the-repository-
    root case), there is no single directory to rename, so each file is
    renamed on its own. An interruption there can leave a partial set,
    which is exactly why the move's job record exists: to say so rather
    than pretend it did not happen.
    """
    tops = {relative.parts[0] if len(relative.parts) > 1 else None
            for relative in relatives}
    top = next(iter(tops)) if len(tops) == 1 else None
    if top is not None and not (target_path / top).exists():
        os.replace(staging / top, target_path / top)
        return
    for relative, destination in zip(relatives, destinations, strict=True):
        destination.parent.mkdir(parents=True, exist_ok=True)
        (staging / relative).replace(destination)


# Checked between chunks during the copy itself, not only once per file — a
# single-file GGUF, the common case, could otherwise not be stopped once
# copying began.
COPY_CHUNK = 64 * 1024 * 1024


@dataclass(frozen=True)
class MoveMechanics:
    """How a move copies, checks and publishes files; a test hands in a failing one."""

    checksum: Callable[[Path], str] = sha256_of
    publish: Callable[[Path, Path, list[Path], list[Path]], None] = publish_staged
    copy_chunk: int = COPY_CHUNK


@dataclass(frozen=True)
class StorageParts:
    """What `ModelStorageService` works with (`move_jobs` makes its job store)."""

    store: ConfigStore
    catalog: Catalog
    host: Host
    on_change: Callable[[str], None]
    writable_in_root: Callable[[Repository, ModelRoot], Repository]
    move_jobs: MoveJobsFactory = MoveJobStore
    mechanics: MoveMechanics = MoveMechanics()


class ModelStorageService:
    """Deleting a model's files, and moving them between storage tiers safely.

    A move copies into a staging area, verifies every file by checksum,
    publishes the copy under its real name, repoints stopped entries, and only
    then removes the source. Every step is recorded in a `move_jobs.MoveJobStore` it holds
    so a crash, a failure or a cancellation is reported rather than lost.
    """

    def __init__(self, parts: StorageParts) -> None:
        self.store = parts.store
        self.catalog = parts.catalog
        self.host = parts.host
        self._changed = parts.on_change
        self._writable_in_root = parts.writable_in_root
        self.jobs = parts.move_jobs(parts.host, parts.on_change)
        self._mechanics = parts.mechanics

    def delete_model(self, model_id: str) -> dict:
        """Remove a model's files from disk.

        Refused while any configured entry points at it — deleting the files
        under a running model would leave a process serving weights that no
        longer exist, and under a stopped one an entry that can never start.
        Removing the entry first is one click, and it makes the order of events
        the user's decision rather than a surprise.

        Every path is checked against the configured repositories before
        anything is unlinked. The model id arrives over HTTP, and this is the
        one operation in the application that destroys data.
        """
        config = self.store.load()
        model = self.catalog.find(config.repositories, model_id)
        users = [item.id for item in config.instances if item.model_id == model_id]
        if users:
            raise ValueError(
                "Remove the entry from the Models tab first: "
                + ", ".join(users) + " still points at this model.")

        roots = [Path(item.path).resolve() for item in config.repositories]
        paths = _inside(roots, [Path(item.path).resolve() for item in model.files])
        freed = sum(item.size_bytes for item in model.files)
        for path in paths:
            path.unlink(missing_ok=True)
        prune_empty(paths, roots)
        self._changed("models")
        return {"deleted": model.name, "files": len(paths), "freed_bytes": freed}

    def move_model(self, model_id: str, target_root_id: str) -> dict:
        """Copy, verify and only then remove a model from its current tier.

        A durable job record is written before anything is touched, and
        updated at every phase (`copying`, `verifying`, `publishing`,
        `completed`/`failed`/`cancelled`). If the process dies partway, that
        record is still on disk — `MoveJobStore.recover()` sweeps it at the next
        startup and marks it failed, rather than leaving a job that claims to
        still be running with no thread behind it. A move can also be
        stopped in flight with `MoveJobStore.cancel(job_id)`, which is checked
        between chunks while a file copies and at each phase boundary.
        """
        config = self.store.load()
        model = self.catalog.find(config.repositories, model_id)
        source_repository = config.repository(model_id.split("/", 1)[0])
        if source_repository.root_id == target_root_id:
            return {"model_id": model_id, "storage_tier": target_root_id,
                    "moved": False}
        plan = self._plan_move(config, model, model_id, source_repository,
                               target_root_id)
        job = self.jobs.open(plan)
        if self._copy_and_publish(plan, job):
            return {"model_id": model_id, "job_id": job["id"],
                    "moved": False, "cancelled": True}
        try:
            record = self._retire_source(config, plan, job)
        except Exception as error:
            # Publishing has succeeded: both complete copies stay, but the
            # durable record must not claim a worker is still active.
            self.jobs.finish(job, "failed", str(error))
            raise
        self.jobs.finish(job, "completed")
        self._changed("models")
        return {**record, "moved": True, "job_id": job["id"]}

    def _copy_and_publish(self, plan: MovePlan, job: dict) -> bool:
        """Copy, verify and publish; True when an operator cancelled the move.

        A cancellation is an operator decision that succeeded, so its staged
        bytes are removed ("temporary files gone on success, cancellation,
        timeout and recovery after restart"). A failure deliberately keeps
        `staging`, published or not: a half-finished move is a resumable
        failure, not garbage, and deleting it once turned an interruption into
        data that existed nowhere. Staging lives outside the scanned tree (see
        `MoveJobStore.open`), so it never shows in the Library as a broken model.
        """
        staging = Path(job["staging"])
        try:
            self._stage_and_publish(plan, job, staging)
        except MoveCancelledError:
            shutil.rmtree(staging, ignore_errors=True)
            self.jobs.finish(job, "cancelled", "Cancelled")
            return True
        except Exception as error:
            self.jobs.finish(job, "failed", str(error))
            raise
        # Everything is published under its real name; staging is empty.
        shutil.rmtree(staging, ignore_errors=True)
        return False

    def _plan_move(self, config, model, model_id: str,
                   source_repository: Repository, target_root_id: str) -> MovePlan:
        """Every check that can refuse a move, before anything is written."""
        self._reject_if_busy(config, model_id)
        self._reject_if_moving(model_id)
        target_root = config.model_root(target_root_id)
        target_repository = self._target_repository(config, source_repository,
                                                     target_root)
        source_root = Path(source_repository.path).resolve()
        target_path = Path(target_repository.path).resolve()
        if source_root == target_path:
            raise ValueError(
                "Source and destination are the same location; refusing to "
                "move a model onto itself")
        sources = [Path(item.path).resolve() for item in model.files]
        reject_unremovable_sources(sources)
        relatives = [path.relative_to(source_root) for path in sources]
        plan = MovePlan(model=model, model_id=model_id,
                         source_repository=source_repository,
                         target_repository=target_repository,
                         target_root=target_root, source_root=source_root,
                         target_path=target_path, sources=sources,
                         relatives=relatives,
                         destinations=[target_path / item for item in relatives])
        reject_unfit_destination(plan)
        return plan

    def _reject_if_moving(self, model_id: str) -> None:
        active = [job for job in self.jobs.all()
                  if job.get("model_id") == model_id
                  and job.get("status") in UNFINISHED_MOVE_STATUSES]
        if active:
            raise ValueError(
                f"{model_id} is already being moved (job {active[0]['id']})")

    def _target_repository(self, config, source_repository: Repository,
                           target_root: ModelRoot) -> Repository:
        """The writable repository in the target tier that mirrors the source."""
        if not target_root.enabled:
            raise ValueError(f"{target_root.name} storage is disabled")
        candidates = [item for item in config.repositories
                      if item.root_id == target_root.id
                      and item.base_id == source_repository.base_id]
        if not candidates:
            raise ValueError(
                f"No {target_root.name} repository matches "
                f"{source_repository.name}")
        return self._writable_in_root(candidates[0], target_root)

    def _stage_and_publish(self, plan: MovePlan, job: dict, staging: Path) -> None:
        """Copy and verify into staging, publish, then repoint stopped entries."""
        self.jobs.phase(job, "copying")
        staging.mkdir(parents=True, exist_ok=True)
        os.chmod(staging, 0o700)
        for source, relative in zip(plan.sources, plan.relatives, strict=True):
            self.jobs.check_cancelled(job)
            staged = staging / relative
            staged.parent.mkdir(parents=True, exist_ok=True)
            self._copy_checking_cancellation(source, staged, job)
            if self._mechanics.checksum(source) != self._mechanics.checksum(staged):
                raise ValueError(
                    f"Checksum mismatch while copying {source.name}")
        self.jobs.phase(job, "verifying")
        self.jobs.check_cancelled(job)
        # Re-checked here, not only at entry: the copy above can run for
        # a long time, and nothing before this point stops a new
        # instance being pointed at the model, or it being loaded, while
        # the copy is in flight.
        self._reject_if_busy(self.store.load(), plan.model_id)
        self.jobs.phase(job, "publishing")
        self._mechanics.publish(staging, plan.target_path, plan.relatives, plan.destinations)
        # A stopped model entry can follow the weights safely. Keeping the
        # assignment is the point of moving storage; forcing somebody to
        # delete and recreate it turns a physical move into configuration
        # loss. This happens only after the verified copy is visible and
        # before the source is removed, so a failed config write leaves
        # both copies rather than a broken entry.
        with self.store.mutate() as live:
            for instance in live.instances:
                if instance.model_id == plan.model_id:
                    instance.model_id = job["target_model_id"]

    def _retire_source(self, config, plan: MovePlan, job: dict) -> dict:
        """Remove the source copy and append the move to the history file.

        Only once every file is verified in the staging area and published
        does the source get touched. A companion file (e.g. a tokenizer)
        can belong to more than one GGUF model in the same directory — see
        `Catalog._classify` — so one still needed by a sibling that has not
        moved is left where it is.
        """
        protected = self._companions_still_needed(config, plan.model, plan.sources)
        removable = [source for source in plan.sources if source not in protected]
        for source in removable:
            source.unlink()
        prune_empty(removable, [plan.source_root])
        record = {"at": time.time(), "source_model_id": plan.model_id,
                  "target_model_id": job["target_model_id"],
                  "source_tier": plan.source_repository.root_id,
                  "target_tier": plan.target_root.id,
                  "bytes": plan.model.size_bytes, "files": len(plan.sources)}
        history = self.host.state_dir() / "model-moves.jsonl"
        with history.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        return record

    def _reject_if_busy(self, config, model_id: str) -> None:
        """Refuse a model that is loaded or loading.

        Called both before the copy starts and again right before publish,
        since the copy can take long enough for that to become true in between.
        Stopped entries are repointed after the verified copy is published.
        """
        loaded = [item.id for item in config.instances
                  if item.model_id == model_id and self.host.status(item.id).running]
        if loaded:
            raise ValueError(
                "This model is currently loaded by " + ", ".join(loaded)
                + ". Unload that entry in Models, then try the move again.")

    def _companions_still_needed(self, config, model, sources: list[Path]) -> set[Path]:
        """Companion files this move must not remove from the source.

        A tokenizer or config file sitting in a GGUF directory is attached by
        the catalog to *every* model in that directory, not just the one
        being moved (`Catalog._classify` groups by directory, not by model).
        Moving one model must not delete a file a sibling still needs to
        load.
        """
        siblings = [item for item in self.catalog.scan(config.repositories)
                    if item.id != model.id]
        source_set = set(sources)
        needed: set[Path] = set()
        for sibling in siblings:
            sibling_paths = {Path(item.path).resolve() for item in sibling.files}
            needed |= sibling_paths & source_set
        return needed

    def _copy_checking_cancellation(self, source: Path, destination: Path,
                                    job: dict) -> None:
        with source.open("rb") as read_from, destination.open("wb") as write_to:
            while True:
                self.jobs.check_cancelled(job)
                chunk = read_from.read(self._mechanics.copy_chunk)
                if not chunk:
                    break
                write_to.write(chunk)
        shutil.copystat(source, destination)



def _inside(roots: list[Path], paths: list[Path]) -> list[Path]:
    """`paths`, after checking that every one is under one of the repositories.

    The model id arrives over HTTP, and deleting is the one operation in the
    application that destroys data.
    """
    for path in paths:
        if not any(path.is_relative_to(root) for root in roots):
            raise ValueError(f"Refusing to delete outside the repositories: {path}")
    return paths
