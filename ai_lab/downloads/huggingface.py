"""Reading the Hugging Face API.

Only search and file listing: enough to find a model and work out which files
it consists of. Uploading, authentication and private repositories are out of
scope.

The grouping applied to a remote listing is the same one the catalog applies
to a local directory — both come from `naming.py` — which is why a model
downloads as one set and then appears in the library as one model.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field

from ..naming import (WEIGHT_SUFFIXES, base_name, is_companion, is_weight,
                      missing_shards)
from .bundles import Bundle

API = "https://huggingface.co/api"
FILES = "https://huggingface.co/{repo}/resolve/main/{path}"
USER_AGENT = "AI-Lab/0.2"
TIMEOUT = 20.0


@dataclass(frozen=True, slots=True)
class RemoteFile:
    """One file upstream.

    `repo` is normally empty, meaning "the repository this set came from". It
    is filled in only for a bundle, whose parts may come from more than one.

    `sha256` is what Hugging Face publishes for a large file. Every weight file
    is stored there through Git LFS, and an LFS entry records the SHA-256 of
    its contents, so the expected hash arrives with the listing rather than
    being computed from the copy that was just downloaded — which would only
    prove the file matches itself. Small text files beside the weights are
    plain Git objects with no such hash; those are checked by size alone, and
    the download report says which check each file got.
    """

    path: str
    size_bytes: int
    repo: str = ""
    sha256: str = ""

    def url(self, repo: str) -> str:
        return FILES.format(repo=self.repo or repo,
                            path=urllib.parse.quote(self.path))


@dataclass(frozen=True, slots=True)
class RemoteSet:
    """One downloadable model: its weight files plus everything beside them."""

    repo: str
    name: str
    format: str
    files: tuple[RemoteFile, ...]
    complete: bool = True
    missing: tuple[str, ...] = field(default=())
    # Filled in for a bundle: what job it is for, and what part of the model
    # each file is. Ordinary sets leave both empty and are unaffected.
    task: str = ""
    roles: dict = field(default_factory=dict)

    @property
    def size_bytes(self) -> int:
        return sum(item.size_bytes for item in self.files)

    def json(self) -> dict:
        payload = {"repo": self.repo, "name": self.name, "format": self.format,
                   "size_bytes": self.size_bytes, "complete": self.complete,
                   "missing": list(self.missing),
                   "files": [{"path": item.path, "size_bytes": item.size_bytes,
                              "repo": item.repo or self.repo,
                              "sha256": item.sha256}
                             for item in self.files]}
        if self.roles:
            payload["task"] = self.task
            payload["roles"] = dict(self.roles)
            payload["bundle"] = True
        return payload


class HuggingFaceClient:
    def __init__(self, opener=None) -> None:
        self._open = opener or self._fetch

    def search(self, query: str, limit: int = 20) -> list[dict]:
        if not query.strip():
            return []
        params = urllib.parse.urlencode(
            {"search": query, "limit": limit, "sort": "downloads", "direction": -1})
        payload = self._open(f"{API}/models?{params}")
        return [{"repo": item.get("modelId") or item.get("id", ""),
                 "downloads": item.get("downloads", 0),
                 "likes": item.get("likes", 0),
                 "updated": item.get("lastModified", ""),
                 "formats": _formats(item)}
                for item in payload]

    def files(self, repo: str) -> list[RemoteFile]:
        quoted = urllib.parse.quote(repo)
        payload = self._open(f"{API}/models/{quoted}/tree/main?recursive=true")
        files = []
        for item in payload:
            if item.get("type") != "file":
                continue
            lfs = item.get("lfs") or {}
            size = item.get("size") or lfs.get("size") or 0
            files.append(RemoteFile(path=item["path"], size_bytes=int(size),
                                    sha256=str(lfs.get("oid") or "")))
        return files

    def sets(self, repo: str,
             bundles: list[Bundle] | None = None) -> list[RemoteSet]:
        """Group a repository listing into the models it contains.

        One repository often holds the same model at several quantisation
        levels — Q4_K_M and Q8_0 side by side — so a listing becomes several
        downloadable sets, not one.

        Any bundle declared under this repository is listed first, because it
        is the thing somebody actually wants: the individual parts below it are
        each unusable alone.
        """
        declared = [item for item in (bundles or []) if item.repo == repo]
        listings = {repo: self.files(repo)}
        for bundle in declared:
            for other in bundle.repos:
                if other not in listings:
                    listings[other] = self.files(other)
        return bundle_sets(declared, listings) + group(repo, listings[repo])

    @staticmethod
    def _fetch(url: str):
        request = urllib.request.Request(url, headers={
            "Accept": "application/json", "User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as error:
            raise ValueError(f"Hugging Face returned {error.code} for {url}") from None
        except (OSError, ValueError) as error:
            raise ValueError(f"Could not reach Hugging Face: {error}") from None


def _formats(item: dict) -> list[str]:
    """Formats advertised by a Hugging Face search result.

    Search results already carry repository tags.  Keeping that information
    means the UI can hide repositories that have no weights any installed
    engine can load, without opening every result and making another network
    request for each one.
    """
    known = {"gguf", "safetensors", "fp8", "nvfp4", "paddleocr"}
    formats = {str(tag).lower() for tag in item.get("tags", [])}
    # A few older GGUF repositories have the suffix but missed the tag.
    repo = str(item.get("modelId") or item.get("id", "")).lower()
    if "gguf" in repo:
        formats.add("gguf")
    return sorted(formats & known)


def group(repo: str, files: list[RemoteFile]) -> list[RemoteSet]:
    """Turn a flat file listing into complete, downloadable models."""
    by_directory: dict[str, list[RemoteFile]] = {}
    for item in files:
        directory = item.path.rsplit("/", 1)[0] if "/" in item.path else ""
        by_directory.setdefault(directory, []).append(item)

    sets: list[RemoteSet] = []
    for directory, entries in sorted(by_directory.items()):
        weights = [item for item in entries if is_weight(item.path.rsplit("/", 1)[-1])]
        companions = [item for item in entries
                      if is_companion(item.path.rsplit("/", 1)[-1])]
        groups: dict[str, list[RemoteFile]] = {}
        for item in weights:
            groups.setdefault(base_name(item.path.rsplit("/", 1)[-1]), []).append(item)
        for base, shards in sorted(groups.items()):
            names = [item.path.rsplit("/", 1)[-1] for item in shards]
            complete, missing = missing_shards(names)
            lower = names[0].lower()
            model_format = next(
                value for suffix, value in WEIGHT_SUFFIXES.items()
                if lower.endswith(suffix))
            name = f"{directory}/{base}" if directory else base
            if model_format == "paddleocr" and base == "inference":
                name = repo.rsplit("/", 1)[-1]
            sets.append(RemoteSet(
                repo=repo,
                name=name,
                format=model_format,
                files=tuple(sorted(shards + companions, key=lambda item: item.path)),
                complete=complete, missing=missing,
            ))
    return sets


def bundle_sets(bundles: list[Bundle],
                listings: dict[str, list[RemoteFile]]) -> list[RemoteSet]:
    """Match declared bundles against real upstream listings.

    A component whose path is not in its repository does not quietly vanish:
    the set comes back incomplete and names the file that is missing. The
    download queue refuses an incomplete set, so a bundle declared against a
    file that upstream has renamed fails as a sentence rather than as a model
    with a hole in it.
    """
    found = []
    for bundle in bundles:
        files, missing = [], []
        for component in bundle.components:
            listing = listings.get(component.repo) or []
            match = next((item for item in listing
                          if item.path == component.path), None)
            if match is None:
                missing.append(f"{component.repo}:{component.path}")
                continue
            files.append(RemoteFile(path=match.path, size_bytes=match.size_bytes,
                                    repo=component.repo, sha256=match.sha256))
        found.append(RemoteSet(
            repo=bundle.repo, name=bundle.name, format=bundle.format,
            files=tuple(files), complete=not missing, missing=tuple(missing),
            task=bundle.task,
            roles={item.file_name: item.role for item in bundle.components},
        ))
    return found
