"""Caps on one client's calls to the card's text models: tokens in flight, calls at once.

Asked by the owner on 2026-11-24, for sqwad.ai on qwen38-nvfp4: the model's
context pool is shared, and Radio-lab's normal work must never be harmed by an
external client's mistakes. Radio-lab's scheduler keeps its own budget; a
client without caps is not counted here at all.

- **What a call costs.** Its prompt plus its answer limit (`max_tokens`, else
  the gateway's default answer cap), because that is the most room the call can
  come to hold in the pool. The prompt is estimated from its characters (one
  token per `limits.clients.chars_per_token`, 3), before the model sees it: no extra round trip,
  and on the safe side, since English runs about four characters a token and
  code about three.
- **Over a cap the call is refused at once**, with 429 and `Retry-After`
  (`limits.clients.retry_after_s`, 5: calls on the card end in seconds to a
  minute, and a client that retries sooner is only refused again), never
  held in a queue: sqwad gives up on a call after 60 s, so a wait in AI-Lab
  would only turn into a failure on its side. A call that alone is larger than
  the client's cap is refused with 400: no retry can help it.
- **The caps are the owner's settings** (the Access page), kept in one private
  file. What is in flight now is in memory only: a restart of the manager ends
  every call anyway.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from pathlib import Path

from ..config.policy import CHARS_PER_TOKEN, ClientCapLimits
from ..shared.numbers import whole_number
from .errors import RefusalError
from .registry import write_private

# The OpenAI and Anthropic shapes of a text call that goes to the card.
TEXT_PATHS = frozenset({"/v1/chat/completions", "/v1/completions", "/v1/messages"})
ANSWER_FIELDS = ("max_tokens", "max_completion_tokens")
CAPS = ("max_tokens_in_flight", "max_calls_at_once")
LIMITS = {"max_tokens_in_flight": (1, 10_000_000), "max_calls_at_once": (1, 1024)}


class OverClientCapError(RefusalError):
    """The client already has as much in flight as its cap allows (429)."""

    code = "over_client_cap"
    # Read by the server (`api.server.status_for`) before its own table.
    http_status = 429

    def __init__(self, message: str, retry_after_s: int, **facts) -> None:
        super().__init__(message, retry_after_s=retry_after_s, **facts)
        self.headers = {"Retry-After": str(retry_after_s)}


class CallOverClientCapError(RefusalError):
    """One call is larger than the client's whole cap: no wait can let it through (400)."""

    code = "call_over_client_cap"


def prompt_estimate(payload: dict, chars_per_token: float = CHARS_PER_TOKEN) -> int:
    """The prompt's tokens, estimated from the characters of everything but the answer limit."""
    sent = {key: value for key, value in payload.items()
            if key not in ANSWER_FIELDS and key != "model"}
    text = json.dumps(sent, ensure_ascii=False, separators=(",", ":"))
    return int(len(text) / chars_per_token) + 1


def answer_limit(payload: dict, default_cap: int) -> int:
    """The answer limit the call names, else the gateway's default cap."""
    for name in ANSWER_FIELDS:
        value = payload.get(name)
        if isinstance(value, int) and value > 0:
            return value
    return max(int(default_cap), 0)


class ClientCaps:
    """The caps per client (the owner's settings), and what each capped client has in flight."""

    def __init__(self, path: Path, limits: ClientCapLimits | None = None) -> None:
        self.path = Path(path)
        # How a prompt's tokens are estimated, and how long a refused client waits.
        self.limits = limits or ClientCapLimits()
        self._lock = threading.Lock()
        data = json.loads(self.path.read_text() or "{}") if self.path.is_file() else {}
        self._caps: dict[str, dict[str, int]] = dict(data.get("clients", {}))
        # name -> [tokens in flight, calls in flight]
        self._held: dict[str, list[int]] = {}

    # -- the owner's settings ----------------------------------------------

    def of(self, name: str | None) -> dict[str, int]:
        """The caps set for `name`; empty when it has none."""
        with self._lock:
            return dict(self._caps.get(name or "", {}))

    def set(self, name: str, changes: dict) -> dict:
        """Set or clear (null or 0) a client's caps; its new view."""
        unknown = set(changes) - set(CAPS)
        if unknown:
            raise ValueError(f"Unknown cap: {', '.join(sorted(unknown))}")
        with self._lock:
            caps = dict(self._caps.get(name, {}))
            for key in CAPS:
                if key not in changes:
                    continue
                value = changes[key]
                if value in (None, 0, ""):
                    caps.pop(key, None)
                    continue
                low, high = LIMITS[key]
                number = whole_number(value, key)
                if not low <= number <= high:
                    raise ValueError(f"{key} must be between {low} and {high}")
                caps[key] = number
            if caps:
                self._caps[name] = caps
            else:
                self._caps.pop(name, None)
            write_private(self.path, {"clients": self._caps})
        return self.view(name)

    def forget(self, name: str) -> None:
        """A removed client's caps go with it."""
        with self._lock:
            if self._caps.pop(name, None) is not None:
                write_private(self.path, {"clients": self._caps})

    def view(self, name: str) -> dict:
        """A client's caps and what it has in flight now."""
        with self._lock:
            tokens, calls = self._held.get(name, (0, 0))
            return {"caps": dict(self._caps.get(name, {})),
                    "in_flight": {"tokens": tokens, "calls": calls}}

    # -- the calls -----------------------------------------------------------

    def admit(self, name: str | None, path: str, payload, default_cap: int) -> Callable[[], None]:
        """Count a capped client's text call, or refuse it; the function that gives it back.

        Anything that is not a capped client's JSON text call passes uncounted,
        and the function returned does nothing.
        """
        caps = self.of(name)
        if not name or not caps or path not in TEXT_PATHS or not isinstance(payload, dict):
            return _nothing
        prompt = prompt_estimate(payload, self.limits.chars_per_token)
        answer = answer_limit(payload, default_cap)
        token_cap = caps.get("max_tokens_in_flight")
        facts = {"client": name, "call_tokens": prompt + answer, "prompt_estimate": prompt,
                 "answer_limit": answer, "max_tokens_in_flight": token_cap,
                 "max_calls_at_once": caps.get("max_calls_at_once")}
        if token_cap is not None and prompt + answer > token_cap:
            raise CallOverClientCapError(
                f"This call counts {prompt + answer} tokens (prompt about {prompt} + answer "
                f"limit {answer}), more than {name}'s cap of {token_cap} tokens in flight: "
                "lower max_tokens or shorten the prompt", **facts)
        return self._count(name, prompt + answer, (token_cap, caps.get("max_calls_at_once")), facts)

    def _count(self, name: str, cost: int, limits: tuple[int | None, int | None],
               facts: dict) -> Callable[[], None]:
        """Add the call to what `name` holds, or refuse it when that goes over a cap."""
        token_cap, call_cap = limits
        with self._lock:
            held = self._held.setdefault(name, [0, 0])
            over_tokens = token_cap is not None and held[0] + cost > token_cap
            if over_tokens or (call_cap is not None and held[1] + 1 > call_cap):
                why = (f"{held[0]} tokens in flight + {cost} > {token_cap}" if over_tokens
                       else f"{held[1]} calls in flight = cap {call_cap}")
                retry = self.limits.retry_after_s
                raise OverClientCapError(
                    f"{name} is at its cap ({why}); retry after {retry} s", retry,
                    tokens_in_flight=held[0], calls_in_flight=held[1], **facts)
            held[0] += cost
            held[1] += 1
        return self._giver(name, cost)

    def _giver(self, name: str, cost: int) -> Callable[[], None]:
        done = threading.Event()

        def give_back() -> None:
            """Take the call off what the client holds, once only."""
            if done.is_set():
                return
            done.set()
            with self._lock:
                held = self._held.get(name)
                if held is not None:
                    held[0] = max(held[0] - cost, 0)
                    held[1] = max(held[1] - 1, 0)
        return give_back


def _nothing() -> None:
    return None
