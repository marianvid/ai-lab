"""The `providers` section of the configuration, read and checked once.

Shape in `config.json`::

    "providers": {
      "vendors": {
        "anthropic": {"dialect": "claude", "binary": "/opt/ai/cli/bin/claude",
                      "home": "/var/lib/ai-lab/cli/anthropic", "concurrency": 6},
        "openai":    {"dialect": "codex", "binary": "/opt/ai/cli/bin/codex",
                      "home": "/var/lib/ai-lab/cli/openai"}
      },
      "models": {
        "claude/sonnet":       {"vendor": "anthropic", "model": "sonnet"},
        "codex/gpt-5.6-terra": {"vendor": "openai", "model": "gpt-5.6-terra"}
      }
    }

Every limit is optional; the defaults are the values measured in production
(see `docs/decisions/0001-subscription-providers.md`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import PurePosixPath

DIALECTS = ("claude", "codex")


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
    """One subscription: which CLI speaks for it, where it lives, its limits."""

    id: str
    dialect: str
    binary: str
    home: str = ""
    limits: VendorLimits = field(default_factory=VendorLimits)

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
        return cls(id=name, dialect=dialect, binary=binary, home=home,
                   limits=VendorLimits.from_mapping(name, raw))


@dataclass(frozen=True)
class ProviderModel:
    """A name the gateway answers to, and the model its CLI is asked for."""

    name: str
    vendor: str
    model: str


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
    return ProviderModel(name=name, vendor=raw["vendor"], model=model.strip())


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
