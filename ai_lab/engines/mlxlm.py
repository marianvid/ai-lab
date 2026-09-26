"""mlx-lm — Apple's own text engine, for Macs only.

MLX is Apple's library for running models on the graphics part of Apple
silicon. mlx-lm is the text-generation program built on it, and it reads the
`mlx` weight format: a folder with `config.json`, a tokenizer and
`*.safetensors` files written by mlx-lm's conversion tool (the
`mlx-community` models on Hugging Face are this). It does not read GGUF.

It is installed as a Python package in its own environment under the runtime
folder, like the other package-installed engines, so `binary` is that
environment's `python`. The process started is AI-Lab's small launcher,
`ai_lab/text/mlxlm_server.py`, which runs mlx-lm's own server but keeps its
health page saying "not yet" until the weights are in memory. The launcher
explains why.

Two things differ from llama.cpp and are worth knowing before comparing them:

* **No context size to reserve.** llama.cpp reserves the memory for a fixed
  context when it starts and divides it between slots. mlx-lm grows each
  request's memory as the text grows, so there is no "context size" setting
  and no division: one long request and several short ones both just work,
  as long as the machine has the memory.
* **Several requests are decoded together.** "Decoding" is producing the
  answer one word-piece at a time. mlx-lm can advance several answers in the
  same step, which is where extra throughput under load comes from. It does
  this only for models whose cache can be merged; for the others it answers
  one request at a time.

The cache keeps the model's own precision (16-bit). mlx-lm has no option to
shrink it.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..capabilities import IMAGES
from ..hosts.command import which
from ..types import Format, ModelSet, Task
from .base import OPENAI_PATHS, LaunchPlan, ParamSpec, validate
from .probe import http_ok

PARAMS = (
    # -- memory: decided when the model starts ------------------------------
    ParamSpec("decode_concurrency", "Answers produced together", "int", 8,
              minimum=1, maximum=64, group="memory",
              help="How many requests mlx-lm advances in the same step. More "
                   "gives more total words per second under load, at the cost "
                   "of memory for each request's cache. The manager also uses "
                   "this as the number of requests it lets in at once. mlx-lm's "
                   "own default is 32; 8 is kinder to a laptop."),
    ParamSpec("prompt_concurrency", "Prompts read together", "int", 4,
              minimum=1, maximum=64, group="memory",
              help="How many new prompts mlx-lm reads in the same step before "
                   "they join the answers being produced. mlx-lm's own "
                   "default is 8."),
    ParamSpec("prefill_step_size", "Prompt chunk", "int", 2048,
              minimum=32, maximum=65536, group="memory",
              help="How many prompt tokens are read in one go. Larger is "
                   "faster for long prompts and needs more memory at the peak. "
                   "Same idea as llama.cpp's batch size."),
    ParamSpec("prompt_cache_size", "Remembered prompts", "int", 10,
              minimum=0, maximum=1000, group="memory",
              help="How many earlier conversations are kept, so a request that "
                   "starts with the same text does not read it again. Useful "
                   "for chat and agents, which resend the whole conversation "
                   "each turn."),
    ParamSpec("prompt_cache_mb", "Remembered prompts memory", "int", 0,
              minimum=0, maximum=1048576, group="memory",
              help="A ceiling, in megabytes, on the memory those remembered "
                   "prompts may use. 0 means no ceiling beyond the count "
                   "above."),

    # -- generation: defaults a client can override --------------------------
    ParamSpec("max_tokens", "Longest answer", "int", 8192,
              minimum=1, maximum=1048576, group="generation",
              help="Tokens to produce when a request does not say. mlx-lm's own "
                   "default is 512, which cuts off a thinking model before it "
                   "starts its answer."),
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
              help="Whether the model thinks before answering. Auto follows "
                   "the model's own template. A single request can still "
                   "choose, by sending "
                   "\"chat_template_kwargs\": {\"enable_thinking\": false}."),
)

# What the Thinking setting puts in the chat template's `enable_thinking`
# switch when a request does not set it itself. "auto" sets nothing.
THINKING = {"on": True, "off": False}

# Room on top of the weights: the cache for the requests being answered, and
# mlx-lm's working memory. A floor like llama.cpp's figure, not a prediction.
MARGIN_FRACTION = 0.10
MARGIN_MIN_MB = 2048.0


def default_launcher() -> str:
    """Where AI-Lab's launcher for mlx-lm sits in this checkout."""
    return str(Path(__file__).resolve().parents[1] / "text" / "mlxlm_server.py")


class MlxLmEngine:
    id = "mlxlm"
    display_name = "MLX LM"

    def __init__(self, binary: str | None = None,
                 server: str | None = None) -> None:
        # The environment's own python. mlx-lm is never on PATH here: it lives
        # in a versioned environment under the runtime folder.
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
            raise ValueError(f"MLX LM cannot load {model.format.value} models")
        if model.task is not Task.TEXT_GENERATION:
            raise ValueError(f"MLX LM cannot perform {model.task.value}")
        if not model.complete:
            raise ValueError(f"{model.name} is missing {len(model.missing)} shard(s)")
        settings = validate(PARAMS, params)
        argv = [
            self.binary, self.server,
            "--model", model.entrypoint,
            "--host", "0.0.0.0",
            "--port", str(port),
            "--decode-concurrency", str(settings["decode_concurrency"]),
            "--prompt-concurrency", str(settings["prompt_concurrency"]),
            "--prefill-step-size", str(settings["prefill_step_size"]),
            "--prompt-cache-size", str(settings["prompt_cache_size"]),
            "--max-tokens", str(settings["max_tokens"]),
            "--temp", str(settings["temperature"]),
            "--top-p", str(settings["top_p"]),
            "--top-k", str(settings["top_k"]),
            "--min-p", str(settings["min_p"]),
        ]
        if settings["prompt_cache_mb"] > 0:
            argv += ["--prompt-cache-bytes", f"{settings['prompt_cache_mb']}MB"]
        if settings["reasoning"] in THINKING:
            argv += ["--chat-template-args", json.dumps(
                {"enable_thinking": THINKING[settings["reasoning"]]})]
        # No chat page: mlx-lm serves an API and nothing a person can open.
        return LaunchPlan(argv=argv, env={"PYTHONUNBUFFERED": "1",
                                          "HF_HUB_OFFLINE": "1"},
                          health_path="/health", web_ui=False)

    def ready(self, port: int) -> bool:
        """The health page, which here means the weights are in memory.

        mlx-lm on its own answers /health before the model is read. The
        launcher makes it answer 503 until the weights are in, so a 200 here
        is a finished load.
        """
        return http_ok(port, "/health")

    def withholds(self, params: dict) -> frozenset[str]:
        """Pictures, always: mlx-lm loads only the text part of a model.

        Many MLX folders are converted from models that can see, and their
        config.json still says so. mlx-lm drops the picture reader on load,
        so an entry on this engine answers text only whatever the files say.
        """
        return frozenset({IMAGES})

    def needs_mb(self, model: ModelSet, params: dict,
                 card_total_mb: float) -> float:
        """The weights on disk, plus a margin for the cache.

        The weights are loaded whole, so their size is the bulk of it. What
        the requests add depends on how long they are and how many run at
        once, which nobody knows at load time, so a margin stands in for it.
        """
        weights = max(0.0, model.size_bytes / (1024 * 1024))
        return weights + max(MARGIN_MIN_MB, weights * MARGIN_FRACTION)

    def concurrency(self, params: dict) -> int:
        """Answers produced together — the number the manager lets in at once."""
        return max(1, int(validate(PARAMS, params)["decode_concurrency"]))

    def api_paths(self, task: Task = Task.TEXT_GENERATION) -> tuple[str, ...]:
        """The OpenAI shape only. mlx-lm does not serve `/v1/messages`."""
        return OPENAI_PATHS if task is Task.TEXT_GENERATION else ()
