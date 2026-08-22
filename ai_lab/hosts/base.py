"""The interface every host implementation must satisfy.

Implementations: `linux.py` and `darwin.py`.
"""

from __future__ import annotations

from pathlib import Path
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

    def system_memory(self) -> tuple[float, float]:
        """How much of the machine's own memory is in use, and how much there is.

        In megabytes, used first. `(0.0, 0.0)` where it cannot be read, which
        reads as "no answer" rather than as an empty machine.

        Separate from the accelerator: a model split between card and system
        memory lives here, and so does everything an engine keeps outside the
        card. On unified memory the two are the same pool and the accelerator
        reading already covers it.
        """
        ...

    def state_dir(self) -> "Path":
        """Where this machine keeps things the manager must not lose.

        Not configuration — that is a file somebody edits, and it lives
        wherever this installation was told to put it. This is the other kind:
        what was true a moment ago and has to survive a restart.

        A platform difference, so it belongs behind this interface rather than
        as a check on the operating system somewhere above.
        """
        ...

    def statuses(self, instance_ids: list[str]) -> dict[str, ProcessStatus]:
        """The same answer as `status`, for several instances at once.

        Asked whenever the model list is drawn, and by the gateway on every
        request, so on a machine where asking is expensive it is worth asking
        once. A host with nothing to gain may leave this alone and let the
        default ask one at a time.
        """
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
