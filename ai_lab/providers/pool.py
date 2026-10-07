"""The subscription lane of the gateway: names in, OpenAI answers out.

`ProviderPool` is what the front door asks when a request names a
subscription model. It finds the vendor, waits at the vendor's gate, runs the
CLI, and decides from the failure's class what happens next (see `errors.py`):

- rate limit → the vendor pauses, the same model is tried again, at most
  `rate_limit_retries` times;
- fatal → an error at once, carrying the CLI's own words;
- transient → retried after each of `network_waits_s`.

Everything it does is counted in `UsageLog`.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

from . import dialects, prompt
from .dialects import DIALECTS, Dialect, Outcome
from .errors import (
    ErrorKind,
    ProviderRateLimitError,
    ProviderUnavailableError,
    classify,
)
from .gate import VendorGate
from .settings import ProviderModel, ProvidersConfig, Vendor
from .usage import UsageLog

Runner = Callable[[Dialect, Vendor, str, str, str], Outcome]
CHAT_PATH = "/v1/chat/completions"


class ProviderPool:
    """Every configured subscription model, each vendor behind its own gate."""

    def __init__(self, config: ProvidersConfig, usage: UsageLog,
                 runner: Runner = dialects.run,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self.config = config
        self.usage = usage
        self._run = runner
        self._sleep = sleep
        self._gates = {name: VendorGate(vendor.limits)
                       for name, vendor in config.vendors.items()}

    @classmethod
    def from_config(cls, raw: dict | None, state_dir: Path) -> ProviderPool:
        """Built from the `providers` section, counting into `state_dir`."""
        return cls(ProvidersConfig.from_mapping(raw), UsageLog(state_dir))

    def serves(self, name: str) -> bool:
        """Whether this name is a subscription model."""
        return name in self.config.models

    def catalogue(self) -> list[dict]:
        """One row per name, in the shape `GET /v1/models` lists."""
        return [{"id": item.name, "object": "model", "owned_by": item.vendor,
                 "ai_lab": {"provider": item.vendor, "loaded": True, "ready": True,
                            "shapes": [CHAT_PATH], "task": "text-generation"}}
                for item in self.config.models.values()]

    def complete(self, path: str, payload: dict) -> dict:
        """Answer one chat request through the vendor's CLI."""
        if path != CHAT_PATH:
            raise ValueError(f"subscription models answer only {CHAT_PATH}")
        if payload.get("stream"):
            raise NotImplementedError("subscription models do not stream; "
                                      "send the request without stream")
        model = self.config.models[payload["model"]]
        text = prompt.render(payload)
        effort = str(payload.get("reasoning_effort") or "")
        return prompt.completion(model.name, self._answer(model, text, effort))

    def _answer(self, model: ProviderModel, text: str, effort: str) -> str:
        vendor = self.config.vendors[model.vendor]
        gate = self._gates[vendor.id]
        dialect = DIALECTS[vendor.dialect]
        rate_limits = transient = 0
        while True:
            with gate.slot():
                outcome = self._run(dialect, vendor, model.model, text, effort)
            if outcome.succeeded:
                gate.succeeded()
                self.usage.record(vendor.id, model.model)
                return outcome.answer
            kind, label = classify(outcome.errors)
            self.usage.record(vendor.id, model.model, failure=label)
            if kind is ErrorKind.FATAL:
                raise ProviderUnavailableError(f"{model.name}: {_tail(outcome.errors)}")
            if kind is ErrorKind.RATE_LIMIT:
                rate_limits += 1
                if rate_limits > vendor.limits.rate_limit_retries:
                    raise ProviderRateLimitError(
                        f"{model.name}: the {vendor.id} allowance is spent for now "
                        f"({_tail(outcome.errors)})")
                gate.rate_limited()
                continue
            if transient >= len(vendor.limits.network_waits_s):
                raise ProviderUnavailableError(f"{model.name}: {_tail(outcome.errors)}")
            self._sleep(vendor.limits.network_waits_s[transient])
            transient += 1

    def stats(self) -> dict:
        """Per vendor: places, any pause, and today's counts."""
        today = self.usage.today()
        return {name: {**gate.state(), "today": today.get(name, {}),
                       "models": sorted(item.name for item in self.config.models.values()
                                        if item.vendor == name)}
                for name, gate in self._gates.items()}


def _tail(text: str, length: int = 300) -> str:
    """The end of a CLI's message, where the cause usually is."""
    flat = " ".join((text or "").split())
    return flat[-length:] or "no output"
