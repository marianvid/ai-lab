"""The interface every host implementation must satisfy.

Implementations: `linux.py` and `darwin.py`.
"""

from __future__ import annotations

from typing import Protocol

from ..types import AcceleratorSnapshot, Capabilities, ProcessSpec, ProcessStatus


class Host(Protocol):
    """Start and stop engine processes, and read the accelerator.

    Implementations must be safe to call from several threads: the web server
    is threaded, and the telemetry sampler polls the accelerator while a load
    is in progress.
    """

    def capabilities(self) -> Capabilities:
        """What this machine supports. Read once at startup, then cached."""
        ...

    def start(self, spec: ProcessSpec) -> None:
        """Launch an engine process. Returns as soon as it has been started.

        Readiness is a separate question — the process being up does not mean
        the weights are loaded. `runtime` polls the engine's own probe for
        that.
        """
        ...

    def stop(self, instance_id: str) -> None:
        """Stop an engine process and wait for it to be gone."""
        ...

    def status(self, instance_id: str) -> ProcessStatus:
        """Whether the process is currently running."""
        ...

    def accelerator(self, pid: int | None = None) -> AcceleratorSnapshot:
        """Read the accelerator now. Called several times a second.

        When `pid` is given, the snapshot also reports how much of the
        accelerator that one process is holding. That is what a progress bar
        needs: with two models resident, the card total says nothing about
        either of them.
        """
        ...

    def logs(self, instance_id: str, lines: int = 15) -> list[str]:
        """The last lines the engine printed.

        Needed when a load fails: the engine knows exactly why — not enough
        memory, a corrupt file, an unsupported quantisation — and that sentence
        is far more useful than "the process exited".
        """
        ...

    def stop_all(self) -> None:
        """Stop every engine this host started, on shutdown.

        A no-op where something else supervises the engines and they are meant
        to survive a manager restart.
        """
        ...
