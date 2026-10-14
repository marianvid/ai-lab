"""Entry point: read the arguments, build the application, serve."""

from __future__ import annotations

import argparse
import signal
import sys
import threading
from pathlib import Path

from .api.server import serve
from .cpu_models import CpuLane
from .lanes import SideLanes
from .providers import ProviderPool
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

    operations, bus, store, model_gateway = build(arguments.config)
    config = store.load()
    host = arguments.host or config.host
    port = arguments.port or config.port
    # Two lanes of their own beside the card: subscription models (Claude
    # Code, Codex, opencode), counting their usage where the host keeps its
    # state, and processor models (embeddings), logging there too.
    state_dir = operations.host.state_dir()
    lanes = SideLanes(providers=ProviderPool.from_config(config.providers, state_dir),
                      cpu=CpuLane.from_config(config.cpu_models, state_dir))
    if lanes.cpu is not None and lanes.cpu.servers:
        lanes.cpu.start_reaper()

    _install_shutdown(operations, lanes)

    operations.recover_moves()
    _restore_last_model(operations)

    print(f"AI-Lab listening on http://{host}:{port}", flush=True)
    try:
        serve(operations, bus, (host, port), model_gateway, lanes)
    finally:
        lanes.stop_all()
        operations.host.stop_all()


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


def _install_shutdown(operations, lanes: SideLanes) -> None:
    """Stop the engines and the processor models when we are asked to quit.

    Python does not run its exit handlers on SIGTERM, so a plain `kill` would
    otherwise leave engines running and holding their ports. The next manager
    would then find a stranger answering its health probe. Where systemd owns
    the engines this is a no-op, because there they are meant to survive. The
    processor models are the manager's own children and always stop with it.
    """
    def shutdown(_signal, _frame):
        lanes.stop_all()
        operations.host.stop_all()
        sys.exit(0)

    for received in (signal.SIGTERM, signal.SIGINT):
        signal.signal(received, shutdown)


if __name__ == "__main__":
    main()
