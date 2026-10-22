"""Turning an engine's "the text is too long" refusal into numbers a client can act on.

vLLM refuses a request longer than the model's `context_size` with HTTP 400 and
an English sentence. A client that wants to cut the text down — ingest with a
long Chinese article, for one — needs the numbers, not the sentence. This
module reads them out of vLLM's own wording and adds them to the answer under
`ai_lab`, next to the engine's message, which stays as it was.

The wording is vLLM 0.26's (`renderers/params.py`, `v1/engine/input_processor.py`).
A sentence that matches none of the patterns is passed on untouched, so a new
vLLM release can make this silent, never wrong.
"""

from __future__ import annotations

import json
import re
from http import HTTPStatus

from .passthrough import Hooks, Passthrough, Waits
from .passthrough import forward as send

# The gateway forwards through this module, so it needs no second import for
# the same act: these are passthrough's own, handed on unchanged.
__all__ = ["Hooks", "Waits", "explained", "forward", "numbers"]

# "This model's maximum context length is N tokens. However, you requested X
# output tokens and your prompt contains [at least] P input tokens, ..."
_BY_TOKENS = re.compile(
    r"maximum context length is (?P<limit>\d+) tokens\. However, you requested "
    r"(?P<output>\d+) output tokens and your prompt contains (?:at least )?"
    r"(?P<prompt>\d+) input tokens")
# The same, when vLLM refused on characters before counting tokens.
_BY_CHARACTERS = re.compile(
    r"maximum context length is (?P<limit>\d+) tokens\. However, you requested "
    r"(?P<output>\d+) output tokens and your prompt contains (?P<chars>\d+) characters")
# The decoder-prompt sentence: its length, then the model's maximum length.
_BY_PROMPT = re.compile(
    r"prompt \(length (?P<prompt>\d+)\).*?longer than the maximum model length of "
    r"(?P<limit>\d+)")

REASON = "context_too_long"


def numbers(message: str) -> dict | None:
    """The limit and the request's size, read out of vLLM's sentence, or None."""
    found = _BY_TOKENS.search(message)
    if found:
        return {"reason": REASON, "context_size": int(found["limit"]),
                "prompt_tokens": int(found["prompt"]),
                "max_output_tokens": int(found["output"])}
    found = _BY_CHARACTERS.search(message)
    if found:
        return {"reason": REASON, "context_size": int(found["limit"]),
                "prompt_characters": int(found["chars"]),
                "max_output_tokens": int(found["output"])}
    found = _BY_PROMPT.search(message)
    if found:
        return {"reason": REASON, "context_size": int(found["limit"]),
                "prompt_tokens": int(found["prompt"])}
    return None


def explained(answer: Passthrough) -> Passthrough:
    """The same answer, with the numbers added when it is a too-long refusal.

    Only a 400 with a JSON body is looked at; everything else, including a
    streamed answer, goes through as it came. A refusal is a single small
    chunk, so reading it whole costs nothing.
    """
    if answer.status != HTTPStatus.BAD_REQUEST or "json" not in answer.content_type:
        return answer
    body = b"".join(answer.chunks)
    try:
        parsed = json.loads(body)
        error = parsed.get("error") if isinstance(parsed, dict) else None
        message = (error.get("message") if isinstance(error, dict) else error) or ""
        detail = numbers(str(message))
    except ValueError:
        detail = None
    if detail is None:
        return Passthrough(answer.status, answer.content_type, iter([body]), answer.headers,
                           answer.close)
    parsed["ai_lab"] = detail
    return Passthrough(answer.status, answer.content_type,
                       iter([json.dumps(parsed).encode()]), answer.headers, answer.close)


def forward(url: str, payload: dict | bytes, waits: Waits, hooks: Hooks,
            content_type: str = "application/json") -> Passthrough:
    """Passthrough's `forward`, with a too-long refusal explained in numbers."""
    return explained(send(url, payload, waits, hooks, content_type=content_type))
