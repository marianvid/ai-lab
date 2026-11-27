"""The gateway's limits: how long to wait, how many to hold, how big an upload.

One value object instead of eleven constructor arguments. Every field has the
default this machine starts with; the real values come from the configuration
(`wiring.build`), because the right numbers differ between a card that reads
8,400 tokens of prompt in under a second and a Mac running a 70 GB model at 17
tokens a second.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from ..config.policy import (BETWEEN_BYTES_S, DEFAULT_MAX_TOKENS, FIRST_BYTE_S,
                             MAX_WAITING, QUIET_MB, QUIET_POLL_S, QUIET_TIMEOUT_S,
                             SAFETY_S, STALL_S)


# How long to wait for an engine and how many requests to hold, and for an
# engine that reports its own progress (vLLM): a request is cut only when the
# engine as a whole has done no work for `STALL_S` while it had some, and never
# kept longer than `SAFETY_S`. Waiting is not a fault (the owner's decision, 22
# October 2026: the 120 s limit goes; what must not happen is a stuck request
# blocking the queue). `DEFAULT_MAX_TOKENS` is the answer cap the gateway adds
# to a text request that names none. The defaults are owned by
# `config/policy.py`, where the configuration's `gateway` section is read.


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
