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

from . import credentials
from . import dialects
from . import prompt
from .dialects import DIALECTS, Ask, Dialect, Outcome
from .errors import (
    RATE,
    USAGE,
    ErrorKind,
    ProviderCallCancelledError,
    ProviderLoginExpiredError,
    ProviderRateLimitError,
    ProviderUnavailableError,
    ProviderVendorLimitedError,
    classify,
    limit_kind,
    said,
)
from .gate import VendorGate, VendorPausedError
from .limit_book import LimitBook, resets_at, utc
from .logins import LoginBook
from .settings import ProviderModel, ProvidersConfig, Vendor
from .usage import UsageLog

Runner = Callable[[Dialect, Vendor, Ask], Outcome]
CHAT_PATH = "/v1/chat/completions"
# The label a call ended by its client is counted under, beside the failures.
CANCELLED = "cancelled"


class ProviderPool:
    """Every configured subscription model, each vendor behind its own gate."""

    def __init__(self, config: ProvidersConfig, usage: UsageLog,
                 runner: Runner = dialects.run_ask,
                 sleep: Callable[[float], None] = time.sleep,
                 logins: LoginBook | None = None) -> None:
        self.config = config
        self.usage = usage
        # Beside the usage counters, so it survives a restart with them.
        self.logins = logins or LoginBook(usage.path.parent)
        self.limits = LimitBook(usage.path.parent)
        self._run = runner
        self._sleep = sleep
        self._gates = {name: VendorGate(vendor.limits)
                       for name, vendor in config.vendors.items()}
        # A wrong default fails at start-up, not on every request.
        for vendor in config.vendors.values():
            _check_effort(f"providers.vendors.{vendor.id}.default_effort",
                          vendor, vendor.default_effort)

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
                            "shapes": [CHAT_PATH], "task": "text-generation",
                            **self._limits(item)}}
                for item in self.config.models.values()]

    def _limits(self, item: ProviderModel) -> dict:
        """The model's limits, shaped like a local model's `params`.

        `max_sequences` is the vendor's calls at once, and it is shared: every model
        of that vendor draws on the same places. `max_sequences_shared_by` names the
        vendor, so a caller counts them together (a local model's are its own and
        carry no such field). `context_size` and `max_output_tokens` only when the
        owner wrote them down.
        """
        params: dict = {"max_sequences": self.config.vendors[item.vendor].limits.concurrency,
                        "max_sequences_shared_by": item.vendor}
        if item.context_size:
            params["context_size"] = item.context_size
        if item.max_output_tokens:
            params["max_output_tokens"] = item.max_output_tokens
        return {"params": params}

    def complete(self, path: str, payload: dict,
                 still_wanted: Callable[[], bool] | None = None) -> dict:
        """Answer one chat request through the vendor's CLI.

        `still_wanted` turns False once the client has gone (its connection
        closed): a call still waiting leaves without starting, a running CLI
        is ended, and `ProviderCallCancelledError` says so to the front door.

        `"ai_lab": {"hold": false}` in the body asks not to be held while the
        vendor is limited: such a call is refused at once with
        `ProviderVendorLimitedError` (503 `vendor_limited`) instead of
        waiting through the vendor's pause, so the caller's local model can
        answer on the spot. Without it a call is held as before.
        """
        if path != CHAT_PATH:
            raise ValueError(f"subscription models answer only {CHAT_PATH}")
        if payload.get("stream"):
            raise NotImplementedError("subscription models do not stream; "
                                      "send the request without stream")
        model = self.config.models[payload["model"]]
        vendor = self.config.vendors[model.vendor]
        effort = str(payload.get("reasoning_effort") or vendor.default_effort)
        _check_effort(model.name, vendor, effort)
        ask = Ask(model.model, prompt.render(payload), effort, still_wanted)
        answer = self._answer(model, ask, _hold(payload))
        if prompt.wants_json(payload):
            answer = prompt.unfence(answer)
        return prompt.completion(model.name, answer)

    def _answer(self, model: ProviderModel, ask: Ask, hold: bool = True) -> str:
        vendor = self.config.vendors[model.vendor]
        gate = self._gates[vendor.id]
        dialect = DIALECTS[vendor.dialect]
        rate_limits = transient = 0
        while True:
            try:
                with gate.slot(ask.still_wanted, hold):
                    outcome = self._run(dialect, vendor, ask)
            except VendorPausedError as paused:
                raise self._refusal(vendor, paused.retry_in_s) from None
            if self._settled(vendor, model, outcome):
                return outcome.answer
            kind, label = classify(outcome.errors)
            self.usage.record(vendor.id, model.model, failure=label)
            if kind is ErrorKind.LOGIN:
                words = said(outcome.errors)
                self.logins.refused(vendor.id, model.model, words)
                raise ProviderLoginExpiredError(vendor.id, self.config.machine, words)
            if kind is ErrorKind.FATAL:
                raise ProviderUnavailableError(f"{model.name}: {_tail(outcome.errors)}")
            if kind is ErrorKind.RATE_LIMIT:
                rate_limits += 1
                self._limited(vendor, model, label, outcome.errors, hold)
                if rate_limits > vendor.limits.rate_limit_retries:
                    raise ProviderRateLimitError(
                        f"{model.name}: the {vendor.id} allowance is spent for now "
                        f"({_tail(outcome.errors)})")
                continue
            if transient >= len(vendor.limits.network_waits_s):
                raise ProviderUnavailableError(f"{model.name}: {_tail(outcome.errors)}")
            self._rest(vendor.limits.network_waits_s[transient], vendor, ask.still_wanted)
            transient += 1

    def _settled(self, vendor: Vendor, model: ProviderModel, outcome: Outcome) -> bool:
        """True for an answer (counted, login and limit cleared); raises for a cancel."""
        if outcome.cancelled:
            # Counted apart: not a failure of the vendor, so no pause, no retry.
            self.usage.record(vendor.id, model.model, failure=CANCELLED)
            raise ProviderCallCancelledError(
                f"{model.name}: the client left; its CLI was ended")
        if not outcome.succeeded:
            return False
        self._gates[vendor.id].succeeded()
        self.usage.record(vendor.id, model.model)
        self.logins.answered(vendor.id, model.model)
        self.limits.answered(vendor.id, model.model)
        return True

    def _limited(self, vendor: Vendor, model: ProviderModel, label: str, errors: str,
                 hold: bool) -> None:
        """A refusal for a limit: pause the vendor, tell the console, refuse at once if asked."""
        pause = self._gates[vendor.id].rate_limited()
        kind = limit_kind(label)
        words = said(errors)
        strikes = self._gates[vendor.id].state()["rate_limit_strikes"]
        if kind == USAGE or strikes >= vendor.limits.limited_after_strikes:
            limit = self.limits.limited(vendor.id, model.model, kind, words, pause)
        else:
            limit = {"kind": kind, "said": words, "resets_at": resets_at(words),
                     "retry_at": utc(time.time() + pause) if pause > 0 else ""}
        if not hold:
            raise ProviderVendorLimitedError(vendor.id, self.config.machine, limit)

    def _refusal(self, vendor: Vendor, retry_in_s: float) -> ProviderVendorLimitedError:
        """The refusal for a call that may not wait through the vendor's pause."""
        limit = self.limits.state(vendor.id)
        if limit.get("state") != "limited":
            limit = {"kind": RATE, "said": ""}
        limit = {**limit, "retry_at": utc(time.time() + retry_in_s)}
        return ProviderVendorLimitedError(vendor.id, self.config.machine, limit)

    def _rest(self, seconds: float, vendor: Vendor,
              still_wanted: Callable[[], bool] | None) -> None:
        """Wait before a retry, leaving early if the client goes meanwhile."""
        if still_wanted is None:
            self._sleep(seconds)
            return
        while seconds > 0:
            if not still_wanted():
                raise ProviderCallCancelledError("the client left before the retry")
            step = min(seconds, vendor.limits.client_check_s)
            self._sleep(step)
            seconds -= step

    def stats(self) -> dict:
        """Per vendor: places, any pause, and today's counts."""
        today = self.usage.today()
        return {name: {**gate.state(), "today": today.get(name, {}),
                       "default_effort": self.config.vendors[name].default_effort,
                       "sign_in": _sign_in(self.config.vendors[name]),
                       "login": self.logins.state(name),
                       "limit": self.limits.state(name),
                       "models": sorted(item.name for item in self.config.models.values()
                                        if item.vendor == name)}
                for name, gate in self._gates.items()}


def _sign_in(vendor: Vendor) -> dict:
    """How the vendor's CLI signs in, for the page; never a value."""
    return credentials.describe(vendor.env_file, vendor.token_warn_after_days,
                                DIALECTS[vendor.dialect].refused_env)


def _hold(payload: dict) -> bool:
    """Whether the call may be held through a vendor's pause (`"ai_lab": {"hold": ...}`)."""
    options = payload.get("ai_lab") or {}
    if not isinstance(options, dict):
        raise ValueError("ai_lab must be an object of settings")
    hold = options.get("hold", True)
    if not isinstance(hold, bool):
        raise ValueError("ai_lab.hold must be true or false")
    return hold


def _check_effort(where: str, vendor: Vendor, effort: str) -> None:
    """Refuse an effort the vendor's CLI would ignore; it would run at another."""
    accepted = DIALECTS[vendor.dialect].efforts
    if effort and accepted and effort not in accepted:
        raise ValueError(f"{where}: reasoning_effort must be one of "
                         f"{', '.join(accepted)}, not {effort!r}")


def _tail(text: str, length: int = 300) -> str:
    """The end of a CLI's message, where the cause usually is."""
    flat = " ".join((text or "").split())
    return flat[-length:] or "no output"
