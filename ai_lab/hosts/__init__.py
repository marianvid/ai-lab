"""The machine we run on.

Starting and stopping processes, and reading the accelerator, are done
completely differently on Linux (systemd, nvidia-smi) and macOS (plain child
processes, unified memory). This package hides that behind one interface so
the rest of the application contains no operating-system branching.

Use `detect.current_host()` to obtain the right implementation.
"""

from .base import Host
from .detect import current_host

__all__ = ["Host", "current_host"]
