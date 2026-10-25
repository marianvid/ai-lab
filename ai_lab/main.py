"""Entry point: read the arguments, build the application, serve."""

from __future__ import annotations

import argparse
import signal
import sys
import threading
from pathlib import Path

from .api.routes import RouteParts
from .api.server import build_router, serve
from .clients.access import Access
from .gpu_health.service import GpuHealth
from .lanes import SideLanes
from .reservations import ReservationBook, ReservationPolicy
from .wiring import build


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """The command line: a configuration file, and optional listen overrides."""
    parser = argparse.ArgumentParser(description="AI-Lab manager")
    parser.add_argument("--config", type=Path, required=True,
                        help="path to config.json")
    parser.add_argument("--host", help="override the listen address")
    parser.add_argument("--port", type=int, help="override the listen port")
    return parser.parse_args(argv)


def main() -> None:
    """Build the manager from its configuration and serve until stopped."""
    arguments = parse_arguments()

    if not arguments.config.is_file():
        sys.exit(f"No configuration at {arguments.config}")

    manager = build(arguments.config)
    operations, lanes = manager.operations, manager.lanes
    config = manager.store.load()
    host = arguments.host or config.host
    port = arguments.port or config.port
    state_dir = manager.host.state_dir()

    # Card reservations (ADR 0005): kept where the host keeps its state, so a
    # restart restores who held the card; the reaper frees a silent holder.
    book = ReservationBook(ReservationPolicy.from_mapping(config.reservations),
                           state_dir / "reservations.json")
    book.start_reaper()

    # The card's long-term health (docs/gpu-health.md): read-only, in its own
    # thread; where there is no NVIDIA driver it only says so.
    health = GpuHealth.start(config.gpu_health, state_dir, manager.gateway.control.activity)

    _install_shutdown(manager.host, lanes)

    operations.recover_moves()
    _restore_last_model(operations)

    # Who may call (A1–A2): tokens and the web session, kept with the host's state.
    access = Access.in_folder(state_dir, log=_log)

    print(f"AI-Lab listening on http://{host}:{port}", flush=True)
    try:
        parts = _parts(manager, book, health, access)
        serve(build_router(operations, parts), manager.bus, (host, port), access)
    finally:
        health.stop()
        lanes.stop_all()
        manager.host.stop_all()


def _parts(manager, book, health, access) -> RouteParts:
    """What the routes are bound to besides the operations."""
    return RouteParts(gateway=manager.gateway, lanes=manager.lanes, book=book,
                      store=manager.store, image_jobs=manager.image_jobs,
                      media_jobs=manager.media_jobs, gpu_health=health, access=access)


def _log(line: str) -> None:
    """One line in the manager's journal, written at once."""
    print(line, flush=True)


def _restore_last_model(operations) -> None:
    """Put back what was on the card, without making anyone wait for it.

    In the background, because a large model takes the better part of a minute
    to load and the page should answer immediately — showing it loading, which
    is the honest thing, rather than refusing to open until it has.

    It does nothing when a model is already running, which is the ordinary case
    on Linux: systemd owns the engines and they survive a manager restart.
    """
    thread = threading.Thread(target=operations.restore_last, daemon=True,
                              name="restore-last-model")
    thread.start()


def _install_shutdown(host, lanes: SideLanes) -> None:
    """Stop the engines and the processor models when we are asked to quit.

    Python does not run its exit handlers on SIGTERM, so a plain `kill` would
    otherwise leave engines running and holding their ports. The next manager
    would then find a stranger answering its health probe. Where systemd owns
    the engines this is a no-op, because there they are meant to survive. The
    processor models are the manager's own children and always stop with it.
    """
    def shutdown(_signal, _frame):
        lanes.stop_all()
        host.stop_all()
        sys.exit(0)

    for received in (signal.SIGTERM, signal.SIGINT):
        signal.signal(received, shutdown)


if __name__ == "__main__":
    main()
