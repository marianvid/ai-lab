"""What the text engines share: the sampling settings, and the MLX engines' common part.

The four sampling settings (temperature, top-p, top-k, min-p) mean the same on
llama.cpp, mlx-lm and mlx-vlm, so they are described once here and each
engine lists them among its own settings. `MlxTextEngine` is what mlx-lm and
mlx-vlm have in common: the `mlx` weight format, text generation, AI-Lab's
launcher script in the engine's own Python, a margin over the weights, and
the OpenAI request shape.
"""

from __future__ import annotations

from pathlib import Path

from ...hosts.command import which
from ...network import ALL_INTERFACES
from ...types import Format, ModelSet, Task
from ..base import OPENAI_PATHS, LaunchPlan, ParamSpec, validate
from ..isolated import AI_LAB_PACKAGE, MIB

TEMPERATURE = ParamSpec(
    "temperature", "Temperature", "float", 0.8, minimum=0.0, maximum=2.0,
    group="generation", help="Higher is more varied, lower is more predictable.")
TOP_P = ParamSpec(
    "top_p", "Top-p", "float", 0.95, minimum=0.0, maximum=1.0, group="generation",
    help="Consider only the most likely tokens adding up to this "
         "probability. 1.0 disables it.")
TOP_K = ParamSpec(
    "top_k", "Top-k", "int", 40, minimum=0, maximum=1000, group="generation",
    help="Consider only this many candidates. 0 disables it.")
MIN_P = ParamSpec(
    "min_p", "Min-p", "float", 0.05, minimum=0.0, maximum=1.0, group="generation",
    help="Drop candidates below this share of the best one.")
SAMPLING = (TEMPERATURE, TOP_P, TOP_K, MIN_P)

# Room on top of the weights for MLX engines: the cache for the requests
# being answered, and the engine's working memory. A floor, not a prediction.
MARGIN_FRACTION = 0.10
MARGIN_MIN_MB = 2048.0


def flag_argv(settings: dict, flags: tuple[tuple[str, str], ...]) -> list[str]:
    """Each (setting, flag) pair as `flag value` on a command line."""
    return [part for key, flag in flags for part in (flag, str(settings[key]))]


def sampling_argv(settings: dict) -> list[str]:
    """The sampling settings as llama.cpp and mlx-lm spell them on the command line."""
    return ["--temp", str(settings["temperature"]), "--top-p", str(settings["top_p"]),
            "--top-k", str(settings["top_k"]), "--min-p", str(settings["min_p"])]


class MlxTextEngine:
    """The part of mlx-lm and mlx-vlm that is the same (see the module's docstring).

    A subclass sets `id`, `display_name`, `settings` (its `PARAMS`),
    `launcher` (the script under `ai_lab/text/`) and `concurrency_setting`
    (the setting that says how many answers are produced together), and
    writes `_settings_argv`: its settings as command-line options.
    """

    settings: tuple[ParamSpec, ...] = ()
    launcher: str = ""
    concurrency_setting: str = ""
    display_name: str

    def __init__(self, binary: str | None = None, server: str | None = None) -> None:
        # The environment's own python. mlx-lm and mlx-vlm are never on PATH
        # here: they live in a versioned environment under the runtime folder.
        self.binary = binary or which("python") or "python"
        self.server = server or default_launcher(self.launcher)

    def formats(self) -> frozenset[Format]:
        """The `mlx` folders: config.json, a tokenizer and safetensors."""
        return frozenset({Format.MLX})

    def tasks(self) -> frozenset[Task]:
        """Text generation."""
        return frozenset({Task.TEXT_GENERATION})

    def params(self, task: Task = Task.TEXT_GENERATION) -> tuple[ParamSpec, ...]:
        """This engine's settings, for text generation."""
        return self.settings if task is Task.TEXT_GENERATION else ()

    def needs_mb(self, model: ModelSet, params: dict, card_total_mb: float) -> float:
        """The weights on disk, plus a margin for the cache.

        The weights are loaded whole, so their size is the bulk of it. What
        the requests add depends on how long they are and how many run at
        once, which nobody knows at load time, so a margin stands in for it.
        """
        weights = max(0.0, model.size_bytes / MIB)
        return weights + max(MARGIN_MIN_MB, weights * MARGIN_FRACTION)

    def concurrency(self, params: dict) -> int:
        """Answers produced together — the number the manager lets in at once."""
        return max(1, int(validate(self.settings, params)[self.concurrency_setting]))

    def api_paths(self, task: Task = Task.TEXT_GENERATION) -> tuple[str, ...]:
        """The OpenAI shape only (pictures, where read, travel inside it)."""
        return OPENAI_PATHS if task is Task.TEXT_GENERATION else ()

    def plan(self, model: ModelSet, port: int, params: dict) -> LaunchPlan:
        """The launcher for this folder, with the engine's settings on its command line.

        No chat page is offered: mlx-lm and mlx-vlm serve an API and nothing a
        person can open.
        """
        settings = self._checked(model, params)
        argv = [self.binary, self.server, "--model", model.entrypoint,
                "--host", ALL_INTERFACES, "--port", str(port), *self._settings_argv(settings)]
        return LaunchPlan(argv=argv, env={"PYTHONUNBUFFERED": "1", "HF_HUB_OFFLINE": "1"},
                          health_path="/health", web_ui=False)

    def _settings_argv(self, settings: dict) -> list[str]:
        """The engine's own settings as its launcher's command-line options."""
        raise NotImplementedError

    def _checked(self, model: ModelSet, params: dict) -> dict:
        """The settings, validated, after refusing a model this engine cannot run."""
        if model.format is not Format.MLX:
            raise ValueError(f"{self.display_name} cannot load {model.format.value} models")
        if model.task is not Task.TEXT_GENERATION:
            raise ValueError(f"{self.display_name} cannot perform {model.task.value}")
        if not model.complete:
            raise ValueError(f"{model.name} is missing {len(model.missing)} shard(s)")
        return validate(self.settings, params)


def default_launcher(script: str) -> str:
    """Where one of AI-Lab's text launchers sits in this checkout."""
    return str(Path(AI_LAB_PACKAGE) / "text" / script)
