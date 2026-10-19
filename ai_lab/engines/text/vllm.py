"""vLLM — the engine for throughput, on NVIDIA hardware only.

It needs CUDA, so it never becomes available on macOS. `registry._reason` says
as much in the interface rather than letting the engine quietly vanish.

Where llama.cpp serves one request at a time well, vLLM interleaves many. On
this machine that difference was measured at up to seventeen times the
throughput as concurrency rises, against about 1.4 for llama.cpp. It costs
startup time: seconds for llama.cpp, roughly a minute here, because the runtime
reconstructs compiled kernels before it will answer.

On a 32 GB card an unquantised model of any real size will not fit, so the
formats that matter are the ones Blackwell accelerates natively: FP8 and NVFP4.
AWQ and GPTQ are listed because they load, not because they are a good choice
here. See MODEL_STORAGE.md.

Unlike llama.cpp there are no generation defaults to set. vLLM takes
temperature, top-p and the rest from each request, so there is nothing to fix at
startup — which is why every setting below belongs to the memory group.
"""

from __future__ import annotations

from ...catalog.capabilities import IMAGES
from ...types import Format, ModelSet, Task
from ...hosts.command import which
from ..base import (ANTHROPIC_PATHS, OPENAI_PATHS, TRANSCRIPTION_PATHS,
                   LaunchPlan, ParamSpec, validate)
from ..probe import http_ok
from ...network import ALL_INTERFACES

# Every one of these was arrived at by running it on this machine. The help
# text says what it costs, because on a single 32 GB card these settings trade
# against each other and the trade is not obvious.
# Parser and cache names are passed through to the installed vLLM version.
# Fixed lists here prevent a new engine release or a parser plugin from being
# configured until AI-Lab itself is changed. The shared identifier validator
# keeps these names bounded; vLLM validates its own supported choices at load.

PARAMS = (
    ParamSpec("context_size", "Context size", "int", 32768,
              minimum=512, maximum=1048576, group="memory",
              help="Longest prompt plus answer the model will accept. vLLM "
                   "shares one pool of cache across all requests rather than "
                   "splitting it per slot, so this is a ceiling for a single "
                   "request, not a division of the total."),
    ParamSpec("gpu_memory_fraction", "GPU memory share", "float", 0.90,
              minimum=0.10, maximum=0.98, group="memory",
              help="How much of the card vLLM claims at startup. Whatever is "
                   "left after the weights becomes cache for concurrent "
                   "requests. Lower it to leave room for another model on the "
                   "same card."),
    ParamSpec("max_sequences", "Concurrent requests", "int", 32,
              minimum=1, maximum=1024, group="memory",
              help="How many requests may be in flight at once. Raising it is "
                   "what buys throughput. It also has to be lowered for models "
                   "with hybrid or linear attention, which reserve state per "
                   "sequence: without it they run out of memory no matter how "
                   "small the context is."),
    ParamSpec("language_model_only", "Text only", "bool", False, group="memory",
              help="Skip the image and audio parts of a multimodal model. On "
                   "startup vLLM pushes fake pictures and sound through those "
                   "paths to measure them — a self-calibration pass that is "
                   "pure waste for text work. Measured on Gemma-4: startup "
                   "halved, from 117 seconds to about 50, with no loss of "
                   "throughput."),
    ParamSpec("kv_cache_dtype", "Cache precision", "identifier", "auto",
              group="memory",
              help="How the context cache is stored. `auto` keeps it at the "
                   "model's own precision. Anything smaller fits more context "
                   "in the same memory — this is the setting to reach for when "
                   "a context you want will not start. Measured here: "
                   "Qwen3-Coder-30B refused 128k because it wanted 12 GiB of "
                   "cache and had 10.78. The engine names the largest that "
                   "fits when it refuses, so the number to try is in the "
                   "error. `fp8` is native on Blackwell, but it wants the "
                   "FlashInfer kernels: without that package the engine "
                   "refuses to start and says which one is missing, so try it "
                   "and read what comes back rather than assuming it works."),
    ParamSpec("prefix_caching", "Reuse a repeated prompt", "bool", False,
              group="memory",
              help="Keep the start of a prompt between requests, so an "
                   "identical opening is not read again. Worth it for an agent, "
                   "which sends the same instructions before every step — "
                   "Claude Code sends about 108,000 characters of them. "
                   "Measured here on Qwen3-Coder-30B with an 8,400-token "
                   "preamble: 0.78 s the first time, 0.21 s every time after. "
                   "Useless for one-off questions that share nothing, where it "
                   "only spends cache that context would have used."),
    ParamSpec("enforce_eager", "Skip kernel compilation", "bool", False,
              group="memory",
              help="Start without building CUDA graphs. Starting is much "
                   "faster and every answer afterwards is slower, so it is for "
                   "trying a model out rather than for using one. The "
                   "compilation is also cached: measured here, the first vLLM "
                   "start after a reinstall took 241 seconds and later ones "
                   "47, so the cost is paid once per model rather than each "
                   "time."),
    ParamSpec("tool_parser", "Tool calling", "identifier", "",
              group="memory",
              help="Empty means the model cannot call tools. Enter a parser name "
                   "supported by the installed vLLM version. The engine "
                   "checks it when loaded. A nonempty name turns tool calling "
                   "on and says how this model writes a "
                   "call, which differs by model family: qwen3_coder for "
                   "Qwen3-Coder, gemma4 for Gemma-4, glm47 for GLM-4.7. "
                   "Needed by any agent that uses tools — Claude Code refuses "
                   "to start without it, with vLLM's own message about "
                   "--enable-auto-tool-choice. Picking the wrong one is not "
                   "silent: the model answers, and its tool calls arrive as "
                   "text nobody acts on."),
    ParamSpec("reasoning_parser", "Thinking parser", "identifier", "",
              group="memory",
              help="How this model marks its thinking, so vLLM can return it "
                   "apart from the answer (`reasoning_content`). Empty means "
                   "the thinking arrives inside the answer text. Measured here "
                   "on Qwen3.6-35B-A3B without it: the answer to a one-line "
                   "question began with paragraphs of English reasoning and a "
                   "stray </think>. Enter a parser name supported by the "
                   "installed vLLM version: qwen3 for Qwen3 and later, "
                   "gemma4 for Gemma-4. It does not switch thinking on or off; "
                   "a request does that with "
                   "chat_template_kwargs.enable_thinking."),
    ParamSpec("tokenizer_mode", "Tokenizer", "identifier", "auto",
              group="memory",
              help="Which tokenizer reads the model's text and builds its chat "
                   "prompt. `auto` lets vLLM choose and suits almost every "
                   "model. `mistral` uses Mistral's own library, for checkpoints "
                   "that ship `tekken.json`. Measured here on "
                   "Mistral-Small-3.2-24B NVFP4: with `auto` every chat request "
                   "was refused with \"default chat template is no longer "
                   "allowed\", because the tokenizer loaded from tekken.json "
                   "carries no chat template. Enter a mode the installed vLLM "
                   "version supports; the engine checks it when loaded."),
    ParamSpec("compact_json", "Compact forced JSON", "bool", True,
              group="memory",
              help="When a request forces a JSON answer (response_format), "
                   "allow no free whitespace inside it. vLLM's default grammar "
                   "lets a model write any amount of space between JSON "
                   "tokens. Measured here on Qwen3.6-35B-A3B: it sometimes "
                   "opened the JSON and then wrote only blank lines until the "
                   "request timed out (2 of 3 Japanese articles). Compact JSON "
                   "leaves nothing to loop on. Off only to compare."),
)

# vLLM's structured-outputs setting that forbids free whitespace in a forced
# JSON answer. The backend has to be named: vLLM 0.26 refuses to start with
# the switch and its default backend `auto`, since only xgrammar and guidance
# honour it. xgrammar is the one `auto` picks first anyway.
COMPACT_JSON = '{"backend": "xgrammar", "disable_any_whitespace": true}'

# Speech recognition has no text context window, tool parser or text-only
# switch. Showing those settings on an ASR model suggests knobs that either do
# nothing or prevent its audio tower from loading.
TRANSCRIPTION_PARAMS = (
    ParamSpec("gpu_memory_fraction", "GPU memory share", "float", 0.90,
              minimum=0.10, maximum=0.98, group="memory",
              help="How much of the card vLLM may use for this speech model."),
    ParamSpec("max_sequences", "Concurrent recordings", "int", 8,
              minimum=1, maximum=128, group="memory",
              help="How many recordings may be transcribed at once."),
    ParamSpec("enforce_eager", "Skip kernel compilation", "bool", False,
              group="memory",
              help="Start faster for testing, at the cost of slower inference."),
)

# Formats vLLM reads. NVFP4 arrives in two different packagings —
# compressed-tensors and NVIDIA's modelopt — and vLLM reads both, which is
# worth knowing because TensorRT-LLM reads only the second.
FORMATS = frozenset({Format.SAFETENSORS, Format.FP8, Format.NVFP4,
                     Format.AWQ, Format.GPTQ})


def _text_flags(settings: dict) -> list[str]:
    """The command-line flags only a text model takes."""
    flags = ["--max-model-len", str(settings["context_size"])]
    if settings["language_model_only"]:
        flags.append("--language-model-only")
    # `auto` is vLLM's own default, so saying it changes nothing and leaving
    # the flag off keeps the command line to what was chosen.
    if settings["kv_cache_dtype"] != "auto":
        flags += ["--kv-cache-dtype", settings["kv_cache_dtype"]]
    if settings["tokenizer_mode"] != "auto":
        flags += ["--tokenizer-mode", settings["tokenizer_mode"]]
    if settings["prefix_caching"]:
        flags.append("--enable-prefix-caching")
    # The two flags go together. vLLM refuses "auto" tool choice unless it has
    # both, so one setting sets both and there is no way to configure half of
    # it.
    if settings["tool_parser"]:
        flags += ["--enable-auto-tool-choice",
                  "--tool-call-parser", settings["tool_parser"]]
    if settings["reasoning_parser"]:
        flags += ["--reasoning-parser", settings["reasoning_parser"]]
    if settings["compact_json"]:
        flags += ["--structured-outputs-config", COMPACT_JSON]
    return flags


class VllmEngine:
    """vLLM as an AI-Lab engine: text and speech models, many requests at once."""

    id = "vllm"
    display_name = "vLLM"

    def __init__(self, binary: str | None = None) -> None:
        # vLLM is normally installed in its own virtual environment, so the
        # command is rarely on PATH. A configured path is the usual case here,
        # not the exception it is for llama.cpp.
        self.binary = binary or which("vllm") or "vllm"

    def formats(self) -> frozenset[Format]:
        """The model formats vLLM reads."""
        return FORMATS

    def tasks(self) -> frozenset[Task]:
        """Text generation and speech recognition."""
        return frozenset({Task.TEXT_GENERATION, Task.TRANSCRIPTION})

    def params(self, task: Task = Task.TEXT_GENERATION) -> tuple[ParamSpec, ...]:
        """The settings a model of this task offers."""
        return TRANSCRIPTION_PARAMS if task is Task.TRANSCRIPTION else PARAMS

    def plan(self, model: ModelSet, port: int, params: dict) -> LaunchPlan:
        """The `vllm serve` command for this model, port and settings."""
        if model.format not in FORMATS:
            raise ValueError(f"vLLM cannot load {model.format.value} models")
        if model.task not in self.tasks():
            raise ValueError(f"vLLM cannot perform {model.task.value}")
        if not model.complete:
            raise ValueError(f"{model.name} is missing {len(model.missing)} shard(s)")
        settings = validate(self.params(model.task), params)
        argv = [
            self.binary, "serve", model.entrypoint,
            "--served-model-name", model.name,
            "--host", ALL_INTERFACES,
            "--port", str(port),
            "--gpu-memory-utilization", str(settings["gpu_memory_fraction"]),
            "--max-num-seqs", str(settings["max_sequences"]),
        ]
        if model.task is Task.TEXT_GENERATION:
            argv += _text_flags(settings)
        if settings["enforce_eager"]:
            argv.append("--enforce-eager")
        # No chat page: vLLM serves an API and nothing a person can open.
        return LaunchPlan(argv=argv, env={}, health_path="/health", web_ui=False)

    def ready(self, port: int) -> bool:
        """Whether the server answers its health check."""
        return http_ok(port, "/health")

    def refuses(self, params: dict) -> frozenset[str]:
        """Mistral's own tokenizer takes no chat-template arguments.

        Measured on Mistral-Small-3.2-24B NVFP4 with `tokenizer_mode` mistral:
        a request carrying `chat_template_kwargs` was answered 400 "chat_template
        is not supported for Mistral tokenizers"; the same request without it
        answered, strict JSON included.
        """
        if params.get("tokenizer_mode") == "mistral":
            return frozenset({"chat_template_kwargs"})
        return frozenset()

    def withholds(self, params: dict) -> frozenset[str]:
        """"Text only" loads a model that can see without the part that sees."""
        return frozenset({IMAGES}) if params.get("language_model_only") else frozenset()

    def needs_mb(self, model, params: dict, card_total_mb: float) -> float:
        """A share of the whole card, and it is exactly that.

        vLLM claims `gpu_memory_utilization` of the card at startup and holds
        it whether the weights fill it or not — measured on the container,
        gemma-4-26b has 17.5 GB of weights and occupies 28.8 GB at 0.9. So the
        model does not come into it: the setting is the answer.

        Which is why lowering the fraction is the only way to put two vLLM
        models on one card, and why a request can send it through `ai_lab`.
        """
        try:
            fraction = float(params.get("gpu_memory_fraction", 0.90))
        except (TypeError, ValueError):
            fraction = 0.90
        return max(0.0, fraction) * max(0.0, card_total_mb)

    def concurrency(self, params: dict) -> int:
        """Sequences in flight. Where the throughput comes from.

        vLLM interleaves them in the same pass rather than taking turns, which
        is why its throughput rises with concurrency — measured on this machine
        at up to seventeen times, against about 1.4 for llama.cpp.
        """
        return max(1, int(validate(PARAMS, params)["max_sequences"]))

    def api_paths(self, task: Task = Task.TEXT_GENERATION) -> tuple[str, ...]:
        """Both shapes.

        Besides the usual one, vLLM serves `/v1/messages` — the shape a client
        written against Anthropic's own library sends. It is the same models
        answering; only the wording of the request differs. That is also why
        vLLM depends on the `anthropic` package: it borrows the request and
        answer definitions rather than writing them out again.
        """
        if task is Task.TRANSCRIPTION:
            return TRANSCRIPTION_PATHS
        return OPENAI_PATHS + ANTHROPIC_PATHS
