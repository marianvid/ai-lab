"""Subscription models (Claude Code, Codex) served through the gateway.

See `docs/decisions/0001-subscription-providers.md` for why this is a lane of
its own beside the card's scheduler, and `pool.py` for how a call goes.
"""

from .errors import ProviderError, ProviderRateLimitError, ProviderUnavailableError
from .pool import ProviderPool
from .settings import ProvidersConfig

__all__ = [
           "ProviderError",
           "ProviderPool",
           "ProviderRateLimitError",
           "ProviderUnavailableError",
           "ProvidersConfig",
]
