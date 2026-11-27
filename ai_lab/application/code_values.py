"""Values still fixed in AI-Lab's code, shown read-only on the Settings page.

The owner's rule: a value that decides how AI-Lab behaves is a setting, editable
where its bounds are clear and at least visible where they are not. These are
the ones not yet made settings (audit L19, L26, L34, L57, L59, L60, L61, m26):
waits of the engine installs, the speech engines' room and defaults, the card
monitor's fixed periods, the music and speech servers' waits, Canary's language,
the models with their own ComfyUI editor, YuE2 Studio's questions.

Each value is read from its module's source text, not imported: several of
these modules run inside an engine's own process (torch, a studio) and must
not be loaded by the manager, and reading keeps a single owner for each number.
A test holds that every entry still names a constant.
"""

from __future__ import annotations

import ast
import operator
from dataclasses import dataclass
from functools import cache
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent.parent
GROUP = "in_code"
NOTE = ("Fixed in AI-Lab's code ({module}, {name}); shown so it is known. Making it a "
        "setting is a change of AI-Lab: ask ai-lab.")
_OPERATIONS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
               ast.Div: operator.truediv, ast.Pow: operator.pow}


@dataclass(frozen=True)
class CodeValue:
    """One constant: where it is, how the page names it, what it does."""

    module: str          # relative to ai_lab/, e.g. builds/package_install.py
    name: str
    label: str
    unit: str
    help: str

    @property
    def key(self) -> str:
        """The field's key: `code.<module>.<NAME>`."""
        return f"code.{self.module.removesuffix('.py').replace('/', '.')}.{self.name}"


def _number(node: ast.AST) -> object:
    """A literal, arithmetic on literals (7 * 86400.0), or a table's names (its string keys,
    sorted); ValueError for anything else."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Dict) and all(isinstance(key, ast.Constant) and isinstance(key.value, str)
                                          for key in node.keys):
        return sorted(str(key.value) for key in node.keys if isinstance(key, ast.Constant))
    if isinstance(node, ast.BinOp) and type(node.op) in _OPERATIONS:
        return _OPERATIONS[type(node.op)](_number(node.left), _number(node.right))
    raise ValueError("not a literal")


@cache
def value_of(module: str, name: str) -> object:
    """The constant `name` as `module`'s source sets it at its top level; KeyError if it does not."""
    tree = ast.parse((PACKAGE / module).read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == name for target in node.targets):
            return _number(node.value)
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) \
                and node.target.id == name and node.value is not None:
            return _number(node.value)
    raise KeyError(f"{module} sets no {name}")


def _wait(module: str, name: str, label: str, what: str) -> CodeValue:
    return CodeValue(module, name, label, "s", what)


VALUES: tuple[CodeValue, ...] = (
    _wait("builds/package_install.py", "CHECK_TIMEOUT_S", "Install: start check",
          "How long a new environment's start check (importing the engine) may take. Too "
          "small: a big CUDA import fails a good install."),
    _wait("builds/package_install.py", "UV_TIMEOUT_S", "Install: one uv question",
          "How long one question to uv (not the install itself) may take."),
    _wait("builds/package_install.py", "INDEX_TIMEOUT_S", "Install: package index",
          "How long the package index may take to say what is newest."),
    _wait("builds/package_environments.py", "VERSION_QUESTION_S", "Installed version question",
          "How long an environment may take to say its package's version."),
    _wait("builds/source_build.py", "VERIFY_TIMEOUT", "Build: start check",
          "How long a new llama.cpp build may take to print its version."),
    _wait("builds/source_compiled.py", "CHECKOUT_S", "Build: git checkout",
          "How long moving the source to a version may take."),
    _wait("builds/source_checkout.py", "GIT_QUESTION_S", "Build: one git question",
          "How long one git question about the checkout may take."),
    _wait("builds/source_checkout.py", "TAG_LISTING_S", "Build: tag listing",
          "How long listing upstream's tags may take."),
    _wait("builds/gitapps.py", "GIT_QUESTION_S", "ComfyUI: one git question",
          "How long one git question about ComfyUI or a custom node may take."),
    _wait("builds/gitapps.py", "VERIFY_S", "ComfyUI: start check",
          "How long a new ComfyUI release may take to import its server."),
    _wait("changes/frompackages.py", "TIMEOUT_S", "What would change: uv dry run",
          "How long uv may take to say which packages an update would replace."),
    CodeValue("engines/speech/qwentts.py", "HEADROOM_MB", "Qwen3-TTS: room beside the weights",
              "MB", "Memory asked for beyond the weights; it decides what is unloaded to make room."),
    CodeValue("engines/speech/voxcpm.py", "HEADROOM_MB", "VoxCPM: room beside the weights", "MB",
              "Memory asked for beyond the weights; it decides what is unloaded to make room."),
    CodeValue("engines/speech/voxcpm.py", "CFG_VALUE", "VoxCPM: default guidance", "",
              "The guidance a VoxCPM entry uses when its options name none."),
    CodeValue("engines/speech/voxcpm.py", "INFERENCE_TIMESTEPS", "VoxCPM: default steps", "",
              "The steps a VoxCPM entry uses when its options name none."),
    _wait("gpu_health/monitor.py", "ANALYSIS_S", "GPU health: analysis every",
          "How often the readings are analysed into windows and alarms."),
    CodeValue("gpu_health/store.py", "FIVE_MINUTES", "GPU health: folded rows", "s",
              "The length of one folded row of readings kept after the raw ones go."),
    _wait("music/backends/yue2_web_backend.py", "SONG_LIMIT_S", "YuE2: one song, longest",
          "How long one song call may run in the studio."),
    _wait("music/backends/comfy_backend.py", "SONG_LIMIT_S", "ComfyUI music: one song, longest",
          "How long one song call may run in ComfyUI."),
    _wait("music/backends/comfy_backend.py", "CONVERT_S", "ComfyUI music: conversion",
          "How long converting a finished song may take."),
    _wait("music/servers/yue2_server.py", "STUDIO_START_S", "YuE2 Studio: start",
          "How long the studio may take to start."),
    _wait("speech/higgs_server.py", "PLAYGROUND_START_S", "Higgs page: start",
          "How long the Higgs page may take to start."),
    _wait("songs/studio.py", "STUDIO_CALL_S", "YuE2 Studio: one question",
          "How long the studio may take to list its jobs or delete a song."),
    CodeValue("songs/studio.py", "STUDIO_JOBS_LOOKED_AT", "YuE2 Studio: jobs looked at", "",
              "How many of the studio's latest jobs are read to know whether one works on a song."),
    _wait("api/routes/music_windows.py", "CANCEL_CALL_S", "YuE2: cancelling a window call",
          "How long the engine may take to say it is cancelling a window call."),
    CodeValue("audio/server.py", "CANARY_LANGUAGE", "Canary: language when none is named", "",
              "The language Canary transcribes when a request names none. Nemotron is given "
              "ro-RO for ro."),
    CodeValue("comfyui_templates/__init__.py", "_TEMPLATES", "ComfyUI: models with their own editor",
              "", "The entries whose ComfyUI page opens a native editor, by name; a model renamed "
              "or added in the Library gets none."),
    CodeValue("comfyui_templates/__init__.py", "PLATFORM_MODEL", "ComfyUI: editor by platform", "",
              "The entry whose native editor differs on macOS and Linux."),
)


def views() -> list[dict]:
    """Each value as a read-only field of the Settings page."""
    return [{"key": item.key, "group": GROUP, "label": item.label, "unit": item.unit,
             "default": value_of(item.module, item.name), "low": None, "high": None,
             "help": f"{item.help} " + NOTE.format(module=item.module, name=item.name),
             "kind": "fixed", "restart": False, "empty": False}
            for item in VALUES]
