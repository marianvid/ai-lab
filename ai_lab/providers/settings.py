"""The `providers` section of the configuration, read and checked once.

Shape in `config.json`::

    "providers": {
      "vendors": {
        "anthropic": {"dialect": "claude", "binary": "/opt/ai/cli/bin/claude",
                      "home": "/var/lib/ai-lab/cli/anthropic", "concurrency": 6},
        "openai":    {"dialect": "codex", "binary": "/opt/ai/cli/bin/codex",
                      "home": "/var/lib/ai-lab/cli/openai"},
        "opencode":  {"dialect": "opencode",
                      "binary": "/opt/ai/cli/opencode/bin/opencode",
                      "home": "/var/lib/ai-lab/cli/opencode"}
      },
      "models": {
        "claude/sonnet":       {"vendor": "anthropic", "model": "sonnet"},
        "codex/gpt-5.6-terra": {"vendor": "openai", "model": "gpt-5.6-terra",
                                "context_size": 258400},
        "opencode/kimi-k2":    {"vendor": "opencode", "model": "<provider>/<model>"}
      }
    }

For opencode, `model` is a `provider/model` name from opencode's own
configuration in its home; the gateway does not repeat that configuration.

A model may also say its limits, which the model list then shows the way it
shows a local model's (`ai_lab.params`): `context_size` (tokens of prompt and
answer the vendor accepts) and `max_output_tokens` (the longest answer). Both
are optional; the vendor's `concurrency` is shown as `max_sequences`.

Every limit is optional; the defaults are the values measured in production
(see `docs/decisions/0001-subscription-providers.md`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import PurePosixPath

DIALECTS = ("claude", "codex", "opencode")


@dataclass(frozen=True)
class VendorLimits:
    """How hard one vendor's subscription may be driven."""

    concurrency: int = 6
    launch_spacing_s: float = 0.5
    rate_limit_waits_s: tuple[float, ...] = (30.0, 120.0, 300.0)
    rate_limit_retries: int = 3
    network_waits_s: tuple[float, ...] = (5.0, 15.0)
    timeout_s: float = 600.0

    @classmethod
    def from_mapping(cls, name: str, raw: dict) -> VendorLimits:
        """Read the optional limits of one vendor, refusing impossible values."""
        default = cls()
        limits = cls(
            concurrency=_whole(name, "concurrency",
                               raw.get("concurrency", default.concurrency)),
            launch_spacing_s=_seconds(name, "launch_spacing_s",
                                      raw.get("launch_spacing_s", default.launch_spacing_s)),
            rate_limit_waits_s=_waits(name, "rate_limit_waits_s",
                                      raw.get("rate_limit_waits_s",
                                              default.rate_limit_waits_s)),
            rate_limit_retries=_whole(name, "rate_limit_retries",
                                      raw.get("rate_limit_retries",
                                              default.rate_limit_retries), minimum=0),
            network_waits_s=_waits(name, "network_waits_s",
                                   raw.get("network_waits_s", default.network_waits_s)),
            timeout_s=_seconds(name, "timeout_s", raw.get("timeout_s", default.timeout_s)))
        if limits.timeout_s <= 0:
            raise ValueError(f"providers.vendors.{name}.timeout_s must be above 0")
        return limits


@dataclass(frozen=True)
class Vendor:
    """One subscription: which CLI speaks for it, where it lives, its limits.

    `default_effort` is the reasoning effort of a request that names none;
    empty leaves the choice to the CLI.
    """

    id: str
    dialect: str
    binary: str
    home: str = ""
    limits: VendorLimits = field(default_factory=VendorLimits)
    default_effort: str = ""

    @classmethod
    def from_mapping(cls, name: str, raw: dict) -> Vendor:
        """One entry of `providers.vendors`."""
        if not isinstance(raw, dict):
            raise ValueError(f"providers.vendors.{name} must be an object")
        dialect = raw.get("dialect")
        if dialect not in DIALECTS:
            raise ValueError(f"providers.vendors.{name}.dialect must be one of {DIALECTS}")
        binary = _absolute(name, "binary", raw.get("binary"))
        home = _absolute(name, "home", raw["home"]) if raw.get("home") else ""
        default_effort = raw.get("default_effort") or ""
        if not isinstance(default_effort, str):
            raise ValueError(f"providers.vendors.{name}.default_effort must be text")
        return cls(id=name, dialect=dialect, binary=binary, home=home,
                   limits=VendorLimits.from_mapping(name, raw),
                   default_effort=default_effort.strip())


@dataclass(frozen=True)
class ProviderModel:
    """A name the gateway answers to, the model its CLI is asked for, and the
    limits the owner wrote down for it (0 when not given)."""

    name: str
    vendor: str
    model: str
    context_size: int = 0
    max_output_tokens: int = 0


@dataclass(frozen=True)
class ProvidersConfig:
    """Every vendor and every model name, cross-checked."""

    vendors: dict[str, Vendor] = field(default_factory=dict)
    models: dict[str, ProviderModel] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, raw: dict | None) -> ProvidersConfig:
        """The whole section; an absent section means no providers."""
        raw = raw or {}
        if not isinstance(raw, dict):
            raise ValueError("providers must be an object")
        vendors = {name: Vendor.from_mapping(name, item)
                   for name, item in (raw.get("vendors") or {}).items()}
        models = {name: _model(name, item, vendors)
                  for name, item in (raw.get("models") or {}).items()}
        return cls(vendors=vendors, models=models)


def _model(name: str, raw: dict, vendors: dict[str, Vendor]) -> ProviderModel:
    if not isinstance(raw, dict) or raw.get("vendor") not in vendors:
        raise ValueError(f"providers.models.{name} must name a configured vendor")
    model = raw.get("model")
    if not isinstance(model, str) or not model.strip():
        raise ValueError(f"providers.models.{name}.model must be a model name")
    return ProviderModel(name=name, vendor=raw["vendor"], model=model.strip(),
                         context_size=_tokens(name, "context_size", raw),
                         max_output_tokens=_tokens(name, "max_output_tokens", raw))


def _tokens(model: str, key: str, raw: dict) -> int:
    """An optional limit in tokens: a whole number above 0, or 0 when absent."""
    value = raw.get(key, 0)
    if type(value) is not int or value < 0:
        raise ValueError(f"providers.models.{model}.{key} must be a whole number of tokens")
    return value


def _absolute(vendor: str, key: str, value) -> str:
    if not isinstance(value, str) or not PurePosixPath(value).is_absolute():
        raise ValueError(f"providers.vendors.{vendor}.{key} must be an absolute path")
    return value


def _whole(vendor: str, key: str, value, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"providers.vendors.{vendor}.{key} must be a whole number "
                         f"of at least {minimum}")
    return value


def _seconds(vendor: str, key: str, value) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or value < 0:
        raise ValueError(f"providers.vendors.{vendor}.{key} must be a number of seconds")
    return float(value)


def _waits(vendor: str, key: str, value) -> tuple[float, ...]:
    if not isinstance(value, list | tuple) or not value:
        raise ValueError(f"providers.vendors.{vendor}.{key} must be a list of seconds")
    return tuple(_seconds(vendor, key, item) for item in value)
