"""The gateway's limits: how long to wait, how many to hold, how big an upload.

One value object instead of eleven constructor arguments. Every field has the
default this machine starts with; the real values come from the configuration
(`wiring.build`), because the right numbers differ between a card that reads
8,400 tokens of prompt in under a second and a Mac running a 70 GB model at 17
tokens a second.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

# How quiet the card has to be before a new model is loaded, and how long to
# wait for that. The threshold is not zero because a driver keeps a small
# allocation of its own; on this machine an idle card reads about 2 MiB.
QUIET_MB = 512.0
QUIET_TIMEOUT_S = 60.0
QUIET_POLL_S = 0.5

# How long to wait for an engine, and how many requests to hold.
FIRST_BYTE_S = 120.0
BETWEEN_BYTES_S = 30.0
MAX_WAITING = 150

# For an engine that reports its own progress (vLLM): a request is cut only
# when the engine as a whole has done no work for `STALL_S` while it had some,
# and never kept longer than `SAFETY_S`. Waiting is not a fault (the owner's
# decision, 22 October 2026: the 120 s limit goes; what must not happen is a
# stuck request blocking the queue).
STALL_S = 60.0
SAFETY_S = 1800.0
# The answer cap the gateway adds to a text request that names none.
DEFAULT_MAX_TOKENS = 4096


@dataclass(frozen=True)
class Patience:
    """How long the gateway lets an engine that reports its progress keep working.

    `stall_s`: no progress at all for this long while the engine has work
    means the engine is stuck; its requests are cut with "engine stalled".
    `safety_s`: no request is kept longer than this, whatever the engine says.
    `default_max_tokens`: the answer cap added to a text request that has
    none (0: none added).
    """

    stall_s: float = STALL_S
    safety_s: float = SAFETY_S
    default_max_tokens: int = DEFAULT_MAX_TOKENS

    def as_dict(self) -> dict:
        """The three numbers, as the Gateway page and `GET /api/gateway` show them."""
        return asdict(self)


@dataclass(frozen=True)
class GatewayLimits:
    """Waits, queue length and upload limits for one gateway.

    `task_timeouts` overrides the two waits per task, keyed by `Task.value`: a
    slow image workflow and a fast text completion share one gateway but must
    not share one timeout. The upload limits are 0 when not set (no limit).
    """

    quiet_mb: float = QUIET_MB
    quiet_timeout_s: float = QUIET_TIMEOUT_S
    poll_s: float = QUIET_POLL_S
    first_byte_s: float = FIRST_BYTE_S
    between_bytes_s: float = BETWEEN_BYTES_S
    max_waiting: int = MAX_WAITING
    task_timeouts: dict = field(default_factory=dict)
    max_upload_bytes: int = 0
    max_upload_pixels: int = 0
    max_upload_dimension: int = 0
    patience: Patience = field(default_factory=Patience)
