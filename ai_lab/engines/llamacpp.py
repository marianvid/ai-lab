"""llama.cpp — the engine available on both Linux and macOS.

Reads GGUF. For a sharded model only the first shard is passed on the command
line; llama-server discovers the rest by name.

The binary is looked up on PATH rather than hard-coded, because it lands in
different places on the two systems: /usr/local/bin from a source build on the
Linux box, /opt/homebrew/bin from Homebrew on the Mac.
"""

from __future__ import annotations

from pathlib import Path

from ..types import Format, ModelSet
from ..hosts.command import which
from .base import LaunchPlan, ParamSpec, validate
from .probe import http_ok

CACHE_TYPES = ("f16", "q8_0", "q5_1", "q5_0", "q4_1", "q4_0")

# Two kinds of setting, and mixing them would mislead. The "memory" group
# decides how much is reserved on the accelerator when the model starts, so
# changing one means reloading. The "generation" group only sets defaults for
# requests: any client can override them per call, so they save you sending the
# same values every time rather than constraining anything.
PARAMS = (
    # -- memory: reserved when the model starts --------------------------
    ParamSpec("context_size", "Context size", "int", 32768,
              minimum=512, maximum=1048576, group="memory",
              help="Tokens the model can attend to. The cache for these sits on "
                   "the accelerator, so a larger context costs memory."),
    ParamSpec("parallel", "Parallel slots", "int", 1, minimum=1, maximum=16,
              group="memory",
              help="How many requests can be served at once. Each slot gets its "
                   "own share of the context."),
    ParamSpec("cache_type_k", "Key cache type", "choice", "q4_0",
              choices=CACHE_TYPES, group="memory",
              help="Quantisation of the attention key cache. Lower uses less "
                   "memory and costs a little quality."),
    ParamSpec("cache_type_v", "Value cache type", "choice", "q4_0",
              choices=CACHE_TYPES, group="memory",
              help="Quantisation of the attention value cache."),
    ParamSpec("flash_attention", "Flash attention", "bool", True, group="memory",
              help="Faster attention kernel. Leave on unless it misbehaves."),
    ParamSpec("batch_size", "Batch size", "int", 2048, minimum=32, maximum=65536,
              group="memory",
              help="Largest number of tokens processed in one go."),
    ParamSpec("ubatch_size", "Micro-batch size", "int", 512, minimum=32,
              maximum=65536, group="memory",
              help="Physical batch handed to the accelerator at a time."),
    ParamSpec("threads", "CPU threads", "int", -1, minimum=-1, maximum=256,
              group="memory",
              help="Threads for the parts that stay on the CPU. -1 lets "
                   "llama.cpp choose."),

    # -- generation: defaults a client can override ------------------------
    ParamSpec("temperature", "Temperature", "float", 0.8, minimum=0.0, maximum=2.0,
              group="generation",
              help="Higher is more varied, lower is more predictable."),
    ParamSpec("top_p", "Top-p", "float", 0.95, minimum=0.0, maximum=1.0,
              group="generation",
              help="Consider only the most likely tokens adding up to this "
                   "probability. 1.0 disables it."),
    # Three separate controls, and the difference matters. `--reasoning off`
    # is the switch. A budget of 0 does something else entirely: it closes the
    # thinking block straight away, and the model carries on writing the same
    # thoughts as its answer — which reads exactly like thinking that was never
    # turned off.
    ParamSpec("reasoning", "Thinking", "choice", "auto",
              choices=("auto", "on", "off"), group="generation",
              help="Whether the model thinks before answering. Auto follows "
                   "what the model's own template asks for."),
    ParamSpec("reasoning_effort", "Thinking effort", "choice", "default",
              choices=("default", "minimal", "low", "medium", "high", "xhigh", "max"),
              group="generation",
              help="How hard to think, for models that understand the "
                   "distinction. Default leaves it to the model."),
    ParamSpec("reasoning_budget", "Thinking token limit", "int", -1,
              minimum=-1, maximum=1048576, group="generation",
              help="A ceiling on thinking tokens. -1 is no limit. Note that 0 "
                   "does not turn thinking off — it cuts the thinking short "
                   "and the rest arrives as the answer. Use Thinking for that."),
    ParamSpec("top_k", "Top-k", "int", 40, minimum=0, maximum=1000,
              group="generation",
              help="Consider only this many candidates. 0 disables it."),
    ParamSpec("min_p", "Min-p", "float", 0.05, minimum=0.0, maximum=1.0,
              group="generation",
              help="Drop candidates below this share of the best one."),
    ParamSpec("repeat_penalty", "Repeat penalty", "float", 1.0,
              minimum=0.0, maximum=2.0, group="generation",
              help="Discourage repeating the same tokens. 1.0 disables it."),
    ParamSpec("reasoning_format", "Thinking format", "choice", "auto",
              choices=("auto", "none", "deepseek", "deepseek-legacy"),
              group="generation",
              help="Where the model's thoughts appear in the reply."),
)

# Settings that used to exist and are now decided rather than configured. They
# are dropped from a configuration file instead of being rejected, so an
# installation written by an older version keeps working.
OBSOLETE = frozenset({"gpu_layers"})

# The whole model goes on the accelerator or it does not load at all. Splitting
# layers between GPU and CPU means every token crosses the link, and this
# machine's accelerator hangs off OCuLink — the split would be slow enough that
# nobody would want the result. Better to say the model does not fit.
ALL_LAYERS = "999"

# Where the built chat page ends up, relative to the binary. It is a directory
# of static files rather than something embedded in the binary, and the server
# serves it only when told where it is — so without --path there is an API and
# no page, which is what a bare GET / answering 415 was telling us.
UI_DIST = ("../tools/ui/dist", "../../tools/server/public")


class LlamaCppEngine:
    id = "llamacpp"
    display_name = "llama.cpp"

    def __init__(self, binary: str | None = None) -> None:
        # A configured path wins over PATH. Two builds of llama.cpp on one
        # machine is normal — a packaged one and one compiled with the flags
        # you wanted — and PATH order is not a decision anybody made.
        self.binary = binary or which("llama-server") or "llama-server"

    def web_ui(self) -> str | None:
        """The chat page this build ships, if it was built.

        Building it needs npm, so a machine without node produces a server with
        an API and no page. Saying so lets the interface offer a link only when
        there is something on the other end.
        """
        base = Path(self.binary).resolve().parent
        for candidate in UI_DIST:
            path = (base / candidate).resolve()
            if (path / "index.html").is_file():
                return str(path)
        return None

    def formats(self) -> frozenset[Format]:
        return frozenset({Format.GGUF})

    def params(self) -> tuple[ParamSpec, ...]:
        return PARAMS

    def plan(self, model: ModelSet, port: int, params: dict) -> LaunchPlan:
        if model.format is not Format.GGUF:
            raise ValueError(f"llama.cpp cannot load {model.format.value} models")
        if not model.complete:
            raise ValueError(f"{model.name} is missing {len(model.missing)} shard(s)")
        settings = validate(PARAMS, {key: value for key, value in params.items()
                                     if key not in OBSOLETE})
        argv = [
            self.binary,
            "--model", model.entrypoint,
            "--alias", model.name,
            "--host", "0.0.0.0",
            "--port", str(port),
            "--n-gpu-layers", ALL_LAYERS,
            "--jinja",
            "--ctx-size", str(settings["context_size"]),
            "--parallel", str(settings["parallel"]),
            "--cache-type-k", settings["cache_type_k"],
            "--cache-type-v", settings["cache_type_v"],
            "--flash-attn", "on" if settings["flash_attention"] else "off",
            "--batch-size", str(settings["batch_size"]),
            "--ubatch-size", str(settings["ubatch_size"]),
            "--temp", str(settings["temperature"]),
            "--top-p", str(settings["top_p"]),
            "--top-k", str(settings["top_k"]),
            "--min-p", str(settings["min_p"]),
            "--repeat-penalty", str(settings["repeat_penalty"]),
            "--reasoning-budget", str(settings["reasoning_budget"]),
        ]
        if settings["reasoning"] != "auto":
            argv += ["--reasoning", settings["reasoning"]]
        if settings["reasoning_effort"] != "default":
            argv += ["--reasoning-effort", settings["reasoning_effort"]]
        if settings["threads"] > 0:
            argv += ["--threads", str(settings["threads"])]
        if settings["reasoning_format"] != "auto":
            argv += ["--reasoning-format", settings["reasoning_format"]]
        ui = self.web_ui()
        if ui:
            argv += ["--path", ui]
        return LaunchPlan(argv=argv, env={}, health_path="/health", web_ui=bool(ui))

    def ready(self, port: int) -> bool:
        return http_ok(port, "/health")
