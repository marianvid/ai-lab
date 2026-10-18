"""An image request turned into the ComfyUI graph its named profile describes.

A profile names a workflow file (API format) under `workflow_root` and maps
request fields onto that graph's node inputs. Only those fields reach the
graph; the uploaded image goes in as the `__AI_LAB_INPUT__` marker, which the
ComfyUI bridge replaces with the uploaded file's name.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..multipart import MultipartBody

# The fields an image request may carry.
ALLOWED = frozenset({"profile", "model", "prompt", "negative_prompt", "seed",
                     "width", "height", "steps", "cfg_scale", "n", "async",
                     "response_format"})
# Form fields read as numbers or a flag when the request is multipart.
WHOLE_NUMBERS = frozenset({"seed", "width", "height", "steps", "n"})
INPUT_MARKER = "__AI_LAB_INPUT__"
# A mapping is [node id, input name].
MAPPING_PARTS = 2


def render(profile: dict, values: dict, workflow_root: Path) -> dict:
    """The profile's graph with the request's values set on the mapped inputs."""
    graph = _graph(profile, workflow_root)
    for name, mapping in profile.get("inputs", {}).items():
        if name == "image":
            _set(graph, name, mapping, INPUT_MARKER)
        elif name in values:
            _set(graph, name, mapping, values[name])
    for node in graph.values():
        if not isinstance(node, dict) or "class_type" not in node or "inputs" not in node:
            raise ValueError("Profile workflow must use ComfyUI API format")
    return graph


def _graph(profile: dict, workflow_root: Path) -> dict:
    """The profile's workflow file, read; it must lie under `workflow_root`."""
    root = workflow_root.resolve()
    path = (root / str(profile.get("workflow", ""))).resolve()
    if root != path and root not in path.parents:
        raise ValueError("Profile workflow escapes workflow_root")
    graph = json.loads(path.read_text())
    if not isinstance(graph, dict) or not graph:
        raise ValueError("Profile workflow is not an API-format ComfyUI graph")
    return graph


def _set(graph: dict, name: str, mapping, value) -> None:
    """One request field onto the node input its mapping names."""
    if not isinstance(mapping, list) or len(mapping) != MAPPING_PARTS:
        raise ValueError(f"Invalid input mapping for {name}")
    node, key = str(mapping[0]), str(mapping[1])
    if node not in graph or "inputs" not in graph[node] or key not in graph[node]["inputs"]:
        raise ValueError(f"Workflow input {name} targets missing {node}.{key}")
    graph[node]["inputs"][key] = value


def read_request(body) -> tuple[dict, bytes | None]:
    """The request's fields and its uploaded image, from JSON or a multipart form."""
    if isinstance(body, MultipartBody):
        fields = {key: _typed(key, value) for key in ALLOWED
                  if (value := body.field(key)) is not None}
        return fields, body.raw("image") or body.raw("file")
    if not isinstance(body, dict):
        raise ValueError("image request must be JSON or multipart")
    return dict(body), None


def _typed(key: str, value: str) -> int | float | bool | str:
    """A form field as the type the JSON form of the request would carry."""
    if key in WHOLE_NUMBERS:
        return int(value)
    if key == "cfg_scale":
        return float(value)
    if key == "async":
        return value.lower() == "true"
    return value
