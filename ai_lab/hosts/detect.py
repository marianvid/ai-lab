"""Pick the host implementation for the machine we are on.

Kept in its own file so that importing `linux` on a Mac — where its
dependencies are absent — never happens by accident.
"""

from __future__ import annotations

import sys

from .base import Host


def current_host(engines: dict | None = None) -> Host:
    """Build the host for this machine.

    `engines` is the `engines` section of config.json. Only Linux needs it, and
    only for vLLM: it is installed in a virtual environment, so PATH cannot be
    used to tell whether it is there.
    """
    if sys.platform == "darwin":
        from .darwin import DarwinHost

        return DarwinHost()
    from .linux import LinuxHost

    settings = engines or {}
    return LinuxHost(vllm_binary=settings.get("vllm", {}).get("binary"))
