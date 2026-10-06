"""mlx-vlm — Apple's MLX for models that read pictures as well as text. Mac only.

It reads the same `mlx` folders as MLX LM (`mlxlm.py`): `config.json`, a
tokenizer and `*.safetensors`. The difference is what it loads from them.
MLX LM loads only the text part; mlx-vlm also loads the part that turns a
picture into something the model can read (the "vision tower"), when the
folder has one. So an entry here keeps the picture capability its files
claim, and a chat request may carry pictures as `image_url` parts, including
`data:` URLs with the picture inside the request.

It is installed as a Python package in its own environment under the runtime
folder, so `binary` is that environment's `python`. The process started is
AI-Lab's launcher, `ai_lab/text/mlxvlm_server.py`, which runs mlx-vlm's own
server and adjusts each request first: the model name, the thinking switch
and the sampling defaults. The launcher explains why each is needed.

Like mlx-lm it has no context size to reserve: each request's memory grows
with its text. It produces several answers in the same step (its "continuous
batching"), up to the number set here.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..hosts.command import which
from ..types import Format, ModelSet, Task
from .base import OPENAI_PATHS, LaunchPlan, ParamSpec, validate
from .probe import http_json
from ..network import ALL_INTERFACES

PARAMS = (
    # -- memory: decided when the model starts ------------------------------
    ParamSpec("max_num_seqs", "Answers produced together", "int", 8,
              minimum=1, maximum=64, group="memory",
              help="How many requests mlx-vlm advances in the same step. More "
                   "gives more total words per second under load, at the cost "
                   "of memory for each request's cache. The manager also uses "
                   "this as the number of requests it lets in at once. "
                   "mlx-vlm's own default has no limit."),
    ParamSpec("prefill_step_size", "Prompt chunk", "int", 2048,
              minimum=32, maximum=65536, group="memory",
              help="How many prompt tokens are read in one go. Larger is "
                   "faster for long prompts and needs more memory at the peak."),
    ParamSpec("max_kv_size", "Longest conversation", "int", 0,
              minimum=0, maximum=1048576, group="memory",
              help="A ceiling, in tokens, on what one request may hold in "
                   "memory. 0 leaves it to the model's own limit."),
    ParamSpec("vision_cache_size", "Remembered pictures", "int", 20,
              minimum=0, maximum=1000, group="memory",
              help="How many pictures are kept after being read, so the same "
                   "picture sent again in a conversation is not read again."),

    # -- generation: defaults a client can override --------------------------
    ParamSpec("max_tokens", "Longest answer", "int", 8192,
              minimum=1, maximum=1048576, group="generation",
              help="Tokens to produce when a request does not say."),
    ParamSpec("temperature", "Temperature", "float", 0.8, minimum=0.0,
              maximum=2.0, group="generation",
              help="Higher is more varied, lower is more predictable."),
    ParamSpec("top_p", "Top-p", "float", 0.95, minimum=0.0, maximum=1.0,
              group="generation",
              help="Consider only the most likely tokens adding up to this "
                   "probability. 1.0 disables it."),
    ParamSpec("top_k", "Top-k", "int", 40, minimum=0, maximum=1000,
              group="generation",
              help="Consider only this many candidates. 0 disables it."),
    ParamSpec("min_p", "Min-p", "float", 0.05, minimum=0.0, maximum=1.0,
              group="generation",
              help="Drop candidates below this share of the best one."),
    ParamSpec("reasoning", "Thinking", "choice", "auto",
              choices=("auto", "on", "off"), group="generation",
              help="Whether the model thinks before answering. mlx-vlm cannot "
                   "leave this to the model's template: on its own it turns "
                   "thinking off. Auto therefore means on, which is what the "
                   "Qwen 3.6 and Gemma 4 templates do by default and what the "
                   "other engines do. A single request can still choose, by "
                   "sending \"chat_template_kwargs\": "
                   "{\"enable_thinking\": false}."),
)

# The sampling settings the launcher fills into a request that leaves them out.
REQUEST_DEFAULTS = ("temperature", "top_p", "top_k", "min_p")

# Room on top of the weights, as for MLX LM: a floor, not a prediction.
MARGIN_FRACTION = 0.10
MARGIN_MIN_MB = 2048.0


def default_launcher() -> str:
    """Where AI-Lab's launcher for mlx-vlm sits in this checkout."""
    return str(Path(__file__).resolve().parents[1] / "text" / "mlxvlm_server.py")


class MlxVlmEngine:
    id = "mlxvlm"
    display_name = "MLX VLM"

    def __init__(self, binary: str | None = None,
                 server: str | None = None) -> None:
        self.binary = binary or which("python") or "python"
        self.server = server or default_launcher()

    def formats(self) -> frozenset[Format]:
        return frozenset({Format.MLX})

    def tasks(self) -> frozenset[Task]:
        return frozenset({Task.TEXT_GENERATION})

    def params(self, task: Task = Task.TEXT_GENERATION) -> tuple[ParamSpec, ...]:
        return PARAMS if task is Task.TEXT_GENERATION else ()

    def plan(self, model: ModelSet, port: int, params: dict) -> LaunchPlan:
        if model.format is not Format.MLX:
            raise ValueError(f"MLX VLM cannot load {model.format.value} models")
        if model.task is not Task.TEXT_GENERATION:
            raise ValueError(f"MLX VLM cannot perform {model.task.value}")
        if not model.complete:
            raise ValueError(f"{model.name} is missing {len(model.missing)} shard(s)")
        settings = validate(PARAMS, params)
        argv = [
            self.binary, self.server,
            "--model", model.entrypoint,
            "--host", ALL_INTERFACES,
            "--port", str(port),
            "--max-num-seqs", str(settings["max_num_seqs"]),
            "--prefill-step-size", str(settings["prefill_step_size"]),
            "--vision-cache-size", str(settings["vision_cache_size"]),
            "--max-tokens", str(settings["max_tokens"]),
        ]
        if settings["max_kv_size"] > 0:
            argv += ["--max-kv-size", str(settings["max_kv_size"])]
        if settings["reasoning"] != "off":
            argv.append("--enable-thinking")
        for key in REQUEST_DEFAULTS:
            argv += ["--request-default", f"{key}={json.dumps(settings[key])}"]
        # No chat page is offered: mlx-vlm serves an API.
        return LaunchPlan(argv=argv, env={"PYTHONUNBUFFERED": "1",
                                          "HF_HUB_OFFLINE": "1"},
                          health_path="/health", web_ui=False)

    def ready(self, port: int) -> bool:
        """The health page names a loaded model.

        mlx-vlm starts answering only after the model given at start is
        loaded, but if that load fails it answers anyway, with no model. So
        the name, not the answer, is what says the load finished.
        """
        return bool(http_json(port, "/health").get("loaded_model"))

    def needs_mb(self, model: ModelSet, params: dict,
                 card_total_mb: float) -> float:
        """The weights on disk, plus a margin for the cache and the pictures."""
        weights = max(0.0, model.size_bytes / (1024 * 1024))
        return weights + max(MARGIN_MIN_MB, weights * MARGIN_FRACTION)

    def concurrency(self, params: dict) -> int:
        """Answers produced together — the number the manager lets in at once."""
        return max(1, int(validate(PARAMS, params)["max_num_seqs"]))

    def api_paths(self, task: Task = Task.TEXT_GENERATION) -> tuple[str, ...]:
        """The OpenAI shape. Pictures travel inside it, as `image_url` parts."""
        return OPENAI_PATHS if task is Task.TEXT_GENERATION else ()
