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
    settings = engines or {}
    if sys.platform == "darwin":
        from .darwin import DARWIN_ENGINES, DarwinHost

        return DarwinHost(_binaries(settings, DARWIN_ENGINES),
                          comfyui_main=(settings.get("comfyui") or {}).get("comfyui"))
    from .linux import LINUX_ENGINES, LinuxHost

    return LinuxHost(_binaries(settings, LINUX_ENGINES))


def _binaries(settings: dict, engines: tuple[str, ...]) -> dict[str, str | None]:
    """Engine id -> the program its section of config.json names, for these engines."""
    return {engine: (settings.get(engine) or {}).get("binary") for engine in engines}
