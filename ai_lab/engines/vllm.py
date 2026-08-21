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

from ..types import Format, ModelSet
from ..hosts.command import which
from .base import ANTHROPIC_PATHS, OPENAI_PATHS, LaunchPlan, ParamSpec, validate
from .probe import http_ok

# Every one of these was arrived at by running it on this machine. The help
# text says what it costs, because on a single 32 GB card these settings trade
# against each other and the trade is not obvious.
# How the context cache is stored, as vLLM names the choices. From
# `vllm serve --help=all` on the installed build, same as the parser list
# below: what is offered here has to exist there.
KV_CACHE_TYPES = (
    "auto", "bfloat16", "float16", "fp8", "fp8_ds_mla", "fp8_e4m3",
    "fp8_e5m2", "fp8_inc", "fp8_per_token_head", "int4_per_token_head",
    "int8_per_token_head", "nvfp4", "nvfp4_4over6", "turboquant_3bit_nc",
    "turboquant_4bit_nc", "turboquant_k3v4_nc", "turboquant_k8v4",
)

# How a model writes a tool call, as vLLM names the formats it can read. Taken
# from `vllm serve --help=all` on the installed build; the list grows with
# every release, and a name added upstream has to be added here before it can
# be chosen. Empty comes first and means tool calling is off.
#
# There is no free-text setting here on purpose. A parser name is checked
# against this list, so a typo is refused when it is typed rather than
# discovered as an engine that will not start.
TOOL_PARSERS = (
    "", "apertus", "cohere_command3", "cohere_command4", "deepseek_v3",
    "deepseek_v31", "deepseek_v32", "deepseek_v4", "dots", "ernie45",
    "functiongemma", "gemma4", "gigachat3", "glm45", "glm47", "granite",
    "granite-20b-fc", "granite4", "hermes", "hunyuan_a13b", "hy_v3", "inkling",
    "internlm", "jamba", "kimi_k2", "kimi_k3", "lfm2", "ling3", "llama3_json",
    "llama4_json", "llama4_pythonic", "longcat", "mimo", "minicpm5",
    "minimax_m2", "minimax_m3", "mistral", "muse_glimmer", "olmo3", "openai",
    "phi4_mini_json", "poolside_v1", "pythonic", "qwen3_coder", "qwen3_xml",
    "seed_oss", "step3", "step3p5", "xlam",
)

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
    ParamSpec("kv_cache_dtype", "Cache precision", "choice", "auto",
              choices=KV_CACHE_TYPES, group="memory",
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
    ParamSpec("tool_parser", "Tool calling", "choice", "",
              choices=TOOL_PARSERS, group="memory",
              help="Empty means the model cannot call tools. Anything else "
                   "turns tool calling on and says how this model writes a "
                   "call, which differs by model family: qwen3_coder for "
                   "Qwen3-Coder, gemma4 for Gemma-4, glm47 for GLM-4.7. "
                   "Needed by any agent that uses tools — Claude Code refuses "
                   "to start without it, with vLLM's own message about "
                   "--enable-auto-tool-choice. Picking the wrong one is not "
                   "silent: the model answers, and its tool calls arrive as "
                   "text nobody acts on."),
)

# Formats vLLM reads. NVFP4 arrives in two different packagings —
# compressed-tensors and NVIDIA's modelopt — and vLLM reads both, which is
# worth knowing because TensorRT-LLM reads only the second.
FORMATS = frozenset({Format.SAFETENSORS, Format.FP8, Format.NVFP4,
                     Format.AWQ, Format.GPTQ})


class VllmEngine:
    id = "vllm"
    display_name = "vLLM"

    def __init__(self, binary: str | None = None) -> None:
        # vLLM is normally installed in its own virtual environment, so the
        # command is rarely on PATH. A configured path is the usual case here,
        # not the exception it is for llama.cpp.
        self.binary = binary or which("vllm") or "vllm"

    def formats(self) -> frozenset[Format]:
        return FORMATS

    def params(self) -> tuple[ParamSpec, ...]:
        return PARAMS

    def plan(self, model: ModelSet, port: int, params: dict) -> LaunchPlan:
        if model.format not in FORMATS:
            raise ValueError(f"vLLM cannot load {model.format.value} models")
        if not model.complete:
            raise ValueError(f"{model.name} is missing {len(model.missing)} shard(s)")
        settings = validate(PARAMS, params)
        argv = [
            self.binary, "serve", model.entrypoint,
            "--served-model-name", model.name,
            "--host", "0.0.0.0",
            "--port", str(port),
            "--max-model-len", str(settings["context_size"]),
            "--gpu-memory-utilization", str(settings["gpu_memory_fraction"]),
            "--max-num-seqs", str(settings["max_sequences"]),
        ]
        if settings["language_model_only"]:
            argv.append("--language-model-only")
        # `auto` is vLLM's own default, so saying it changes nothing and
        # leaving the flag off keeps the command line to what was chosen.
        if settings["kv_cache_dtype"] != "auto":
            argv += ["--kv-cache-dtype", settings["kv_cache_dtype"]]
        if settings["prefix_caching"]:
            argv.append("--enable-prefix-caching")
        if settings["enforce_eager"]:
            argv.append("--enforce-eager")
        # The two flags go together. vLLM refuses "auto" tool choice unless it
        # has both, so one setting sets both and there is no way to configure
        # half of it.
        if settings["tool_parser"]:
            argv += ["--enable-auto-tool-choice",
                     "--tool-call-parser", settings["tool_parser"]]
        # No chat page: vLLM serves an API and nothing a person can open.
        return LaunchPlan(argv=argv, env={}, health_path="/health", web_ui=False)

    def ready(self, port: int) -> bool:
        return http_ok(port, "/health")

    def api_paths(self) -> tuple[str, ...]:
        """Both shapes.

        Besides the usual one, vLLM serves `/v1/messages` — the shape a client
        written against Anthropic's own library sends. It is the same models
        answering; only the wording of the request differs. That is also why
        vLLM depends on the `anthropic` package: it borrows the request and
        answer definitions rather than writing them out again.
        """
        return OPENAI_PATHS + ANTHROPIC_PATHS
