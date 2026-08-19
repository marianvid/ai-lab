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
from .base import LaunchPlan, ParamSpec, validate
from .probe import http_ok

# Every one of these was arrived at by running it on this machine. The help
# text says what it costs, because on a single 32 GB card these settings trade
# against each other and the trade is not obvious.
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
        # No chat page: vLLM serves an API and nothing a person can open.
        return LaunchPlan(argv=argv, env={}, health_path="/health", web_ui=False)

    def ready(self, port: int) -> bool:
        return http_ok(port, "/health")
