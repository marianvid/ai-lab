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

from ..base import ParamSpec
from ..probe import http_json
from .text_launch import SAMPLING, MlxTextEngine, flag_argv

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
    *SAMPLING,
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
# The settings mlx-vlm's server takes as they are, and its name for each.
FLAGS = (("max_num_seqs", "--max-num-seqs"), ("prefill_step_size", "--prefill-step-size"),
         ("vision_cache_size", "--vision-cache-size"), ("max_tokens", "--max-tokens"))


class MlxVlmEngine(MlxTextEngine):
    """mlx-vlm's own server, behind AI-Lab's launcher (`text/mlxvlm_server.py`)."""

    id = "mlxvlm"
    display_name = "MLX VLM"
    settings = PARAMS
    launcher = "mlxvlm_server.py"
    concurrency_setting = "max_num_seqs"

    def _settings_argv(self, settings: dict) -> list[str]:
        """These settings as the launcher's command-line options."""
        argv = flag_argv(settings, FLAGS)
        if settings["max_kv_size"] > 0:
            argv += ["--max-kv-size", str(settings["max_kv_size"])]
        if settings["reasoning"] != "off":
            argv.append("--enable-thinking")
        for key in REQUEST_DEFAULTS:
            argv += ["--request-default", f"{key}={json.dumps(settings[key])}"]
        return argv

    def ready(self, port: int) -> bool:
        """The health page names a loaded model.

        mlx-vlm starts answering only after the model given at start is
        loaded, but if that load fails it answers anyway, with no model. So
        the name, not the answer, is what says the load finished.
        """
        return bool(http_json(port, "/health").get("loaded_model"))
