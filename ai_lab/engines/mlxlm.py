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

from ..capabilities import IMAGES
from .base import ParamSpec
from .probe import http_ok
from .text_launch import SAMPLING, MlxTextEngine, flag_argv, sampling_argv

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
    *SAMPLING,
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
# The settings mlx-lm's server takes as they are, and its name for each.
FLAGS = (("decode_concurrency", "--decode-concurrency"),
         ("prompt_concurrency", "--prompt-concurrency"),
         ("prefill_step_size", "--prefill-step-size"),
         ("prompt_cache_size", "--prompt-cache-size"),
         ("max_tokens", "--max-tokens"))


class MlxLmEngine(MlxTextEngine):
    """mlx-lm's own server, behind AI-Lab's launcher (`text/mlxlm_server.py`)."""

    id = "mlxlm"
    display_name = "MLX LM"
    settings = PARAMS
    launcher = "mlxlm_server.py"
    concurrency_setting = "decode_concurrency"

    def _settings_argv(self, settings: dict) -> list[str]:
        """These settings as the launcher's command-line options."""
        argv = [*flag_argv(settings, FLAGS), *sampling_argv(settings)]
        if settings["prompt_cache_mb"] > 0:
            argv += ["--prompt-cache-bytes", f"{settings['prompt_cache_mb']}MB"]
        if settings["reasoning"] in THINKING:
            argv += ["--chat-template-args", json.dumps(
                {"enable_thinking": THINKING[settings["reasoning"]]})]
        return argv

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
