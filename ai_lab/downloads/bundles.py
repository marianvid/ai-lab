"""Models that are several files from several places, downloaded as one thing.

Most models are one directory upstream: ask for it and everything you need
comes with it. The image models are not. A ComfyUI image model is assembled
from parts that live in different folders of a repository, and sometimes in
different repositories altogether — the part that draws, the part that reads
the prompt, and the part that turns the result into a picture. Qwen-Image-Edit
is the clear case: its drawing part is in its own repository, and the two other
parts it needs are in the plain Qwen-Image repository next door.

Nothing in a listing says which parts belong together. Guessing would be wrong
often enough to matter — a repository holds the same model at five different
precisions, and pairing the wrong drawing part with the wrong prompt reader
produces a model that loads and makes rubbish. So the grouping is *declared*:
somebody writes down the exact upstream file paths once, and from then on
AI-Lab treats that group as a single downloadable model with a name of its own.

The declaration lives in configuration, under `downloads.bundles`:

    {"name": "qwen-image-2512-nvfp4",
     "repo": "Comfy-Org/Qwen-Image_ComfyUI",
     "format": "comfyui",
     "task": "image-generation",
     "components": [
       {"role": "diffusion_model",
        "path": "split_files/diffusion_models/qwen_image_nvfp4.safetensors"},
       {"role": "text_encoder",
        "path": "split_files/text_encoders/qwen_2.5_vl_7b_nvfp4.safetensors"},
       {"role": "vae", "path": "split_files/vae/qwen_image_vae.safetensors"}]}

`repo` is the repository the bundle is listed under, so that browsing that
repository in the Library shows the bundle beside the individual files. A
component may name a `repo` of its own when the file comes from somewhere else.

This module is pure text: it reads declarations and checks that they are safe
and coherent. Matching them against a real upstream listing belongs to the
client that does the listing, and downloading belongs to the queue.
"""

from __future__ import annotations

from dataclasses import dataclass

# Roles are ComfyUI's own words for the categories it looks a file up under.
# They are recorded for documentation and for the report; the engine finds a
# file by its name, because every part of a bundle lands in one directory and
# ComfyUI is pointed at that directory for every category at once.
ROLES = frozenset({
    "diffusion_model", "text_encoder", "vae", "lora", "controlnet",
    "clip_vision", "checkpoint", "upscale_model",
})


@dataclass(frozen=True, slots=True)
class Component:
    role: str
    repo: str
    path: str

    @property
    def file_name(self) -> str:
        return self.path.rsplit("/", 1)[-1]


@dataclass(frozen=True, slots=True)
class Bundle:
    """One model assembled from named upstream files."""

    name: str
    repo: str
    format: str
    task: str
    components: tuple[Component, ...]

    @property
    def repos(self) -> tuple[str, ...]:
        """Every repository this bundle reads from, the main one first."""
        seen = [self.repo]
        for item in self.components:
            if item.repo not in seen:
                seen.append(item.repo)
        return tuple(seen)


def parse(raw: list[dict] | None) -> list[Bundle]:
    """Read the declarations, refusing anything unsafe or incoherent.

    Refusing here rather than at download time means a mistake in the
    configuration is reported once, in words, instead of becoming a file
    written somewhere it should never have gone.
    """
    bundles = []
    for index, item in enumerate(raw or []):
        where = item.get("name") or f"bundle {index + 1}"
        name = _safe_name(item.get("name", ""), where)
        repo = _safe_repo(item.get("repo", ""), where)
        components = _components(item.get("components", []), repo, where)
        bundles.append(Bundle(
            name=name, repo=repo,
            format=str(item.get("format") or "comfyui"),
            task=str(item.get("task") or "image-generation"),
            components=components,
        ))
    _distinct([item.name for item in bundles])
    return bundles


# -- the rules ------------------------------------------------------------

def _safe_name(value: str, where: str) -> str:
    """The name becomes a directory in the model store, so it must be a name.

    A name with a slash in it would put the model somewhere other than the
    repository it was downloaded into; one starting with a dot would hide it
    from the library that is supposed to show it.
    """
    name = str(value).strip()
    if not name:
        raise ValueError(f"{where}: a bundle needs a name")
    if "/" in name or "\\" in name or name.startswith(".") or ".." in name:
        raise ValueError(
            f"{where}: {name!r} is not a usable name — a bundle's name becomes "
            "a folder in the model store, so it cannot contain a slash, start "
            "with a dot, or step up a directory")
    return name


def _safe_repo(value: str, where: str) -> str:
    repo = str(value).strip().strip("/")
    if repo.count("/") != 1 or not all(repo.split("/")):
        raise ValueError(
            f"{where}: {value!r} is not a repository — expected owner/name")
    if ".." in repo:
        raise ValueError(f"{where}: {value!r} steps outside its own repository")
    return repo


def _safe_path(value: str, where: str) -> str:
    """An upstream path: relative, inside the repository, naming a file."""
    path = str(value).strip()
    if not path:
        raise ValueError(f"{where}: a component needs a path")
    if path.startswith("/") or "\\" in path:
        raise ValueError(
            f"{where}: {path!r} must be a path inside the repository, written "
            "with forward slashes and no leading slash")
    if any(part in ("", ".", "..") for part in path.split("/")):
        raise ValueError(f"{where}: {path!r} steps outside the repository")
    return path


def _components(raw: list, repo: str, where: str) -> tuple[Component, ...]:
    if not raw:
        raise ValueError(f"{where}: a bundle needs at least one component")
    components = []
    for item in raw:
        role = str(item.get("role") or "").strip()
        if role not in ROLES:
            raise ValueError(
                f"{where}: {role!r} is not a part a ComfyUI model is made of. "
                f"Known parts: {', '.join(sorted(ROLES))}")
        components.append(Component(
            role=role,
            repo=_safe_repo(item.get("repo") or repo, where),
            path=_safe_path(item.get("path", ""), where),
        ))
    # Every part lands in one directory, so two parts cannot share a file name:
    # the second would overwrite the first and the model would be missing a
    # piece without anything having failed.
    _distinct([item.file_name for item in components],
              f"{where}: two components are both called ")
    return tuple(components)


def _distinct(names: list[str], prefix: str = "Two bundles are both called ") -> None:
    seen = set()
    for name in names:
        if name in seen:
            raise ValueError(f"{prefix}{name}")
        seen.add(name)
