"""The default answer cap: a text request that names no length limit gets one.

Without `max_tokens` vLLM lets an answer run to the end of the context — on
qwen36-nvfp4, 131,072 tokens, over ten minutes holding a place and a share of
the pool. A request that forgot its cap is the likeliest runaway. So the
gateway adds one (the `default_max_tokens` setting, 4,096 unless changed;
approved by the owner on 22 October 2026) and says so in a response header.

vLLM refuses a request whose prompt plus `max_tokens` exceeds the model's
context ("you requested 4096 output tokens and your prompt contains ...").
Without the cap the same request would have been answered, with whatever room
was left. So when the cap the gateway added is the reason for that refusal,
the request is sent once more without it, under the same card place.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from http import HTTPStatus

from .context_limit import REASON, Hooks, Waits, forward
from .passthrough import Passthrough

# The gateway forwards through this module, so it needs no second import for
# the same act: these are `context_limit`'s own, handed on unchanged.
__all__ = ["HEADER", "Hooks", "PlaceKeeper", "Waits", "add_cap", "forward", "send_capped"]

# The OpenAI shapes a text request arrives in. Anthropic's `/v1/messages`
# requires `max_tokens` itself, so it never arrives without one.
CAPPED_PATHS = frozenset({"/v1/chat/completions", "/v1/completions"})
LIMIT_FIELDS = ("max_tokens", "max_completion_tokens")
# Tells the client the gateway capped its answer, and at how much.
HEADER = "X-AI-Lab-Default-Max-Tokens"


class PlaceKeeper:
    """Gives the card place back when the answer closes — but not while the gateway retries."""

    def __init__(self, release: Callable[[], None]) -> None:
        self._release = release
        self.holding = False

    def __call__(self) -> None:
        """The answer closed: give the place back, unless a retry still needs it."""
        if not self.holding:
            self._release()


def add_cap(request: dict, path: str, cap: int) -> int:
    """Put the cap in when this text request names no limit; the cap added, or 0."""
    if cap <= 0 or path not in CAPPED_PATHS:
        return 0
    if any(request.get(name) is not None for name in LIMIT_FIELDS):
        return 0
    request["max_tokens"] = cap
    return cap


def send_capped(send: Callable[[dict], Passthrough], request: dict, path: str, cap: int,
                keeper: PlaceKeeper) -> Passthrough:
    """Send with the default cap added when due; without it again if it caused a refusal."""
    added = add_cap(request, path, cap)
    if not added:
        return send(request)
    keeper.holding = True
    try:
        answer = send(request)
    finally:
        keeper.holding = False
    too_long, answer = _refused_for_length(answer)
    if too_long:
        del request["max_tokens"]
        return send(request)
    answer.headers[HEADER] = str(added)
    first_close = answer.close

    def close() -> None:
        first_close()
        keeper()
    return Passthrough(answer.status, answer.content_type, answer.chunks, answer.headers, close)


def _refused_for_length(answer: Passthrough) -> tuple[bool, Passthrough]:
    """Whether vLLM refused for length, and the answer, readable again."""
    if answer.status != HTTPStatus.BAD_REQUEST or "json" not in answer.content_type:
        return False, answer
    body = b"".join(answer.chunks)
    try:
        parsed = json.loads(body)
    except ValueError:
        parsed = None
    reason = parsed.get("ai_lab", {}).get("reason") if isinstance(parsed, dict) else None
    again = Passthrough(answer.status, answer.content_type, iter([body]), answer.headers,
                        answer.close)
    return reason == REASON, again
