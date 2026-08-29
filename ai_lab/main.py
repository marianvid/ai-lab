"""Entry point: read the arguments, build the application, serve."""

from __future__ import annotations

import argparse
import signal
import sys
import threading
from pathlib import Path

from .api.server import serve
from .wiring import build

DEFAULT_CONFIG = Path("/etc/ai-lab/config.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="AI-Lab manager")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="path to config.json")
    parser.add_argument("--host", help="override the listen address")
    parser.add_argument("--port", type=int, help="override the listen port")
    arguments = parser.parse_args()

    if not arguments.config.is_file():
        sys.exit(f"No configuration at {arguments.config}")

    operations, bus, store, model_gateway = build(arguments.config)
    config = store.load()
    host = arguments.host or config.host
    port = arguments.port or config.port

    _install_shutdown(operations)

    operations.recover_moves()
    _restore_last_model(operations)

    print(f"AI-Lab listening on http://{host}:{port}", flush=True)
    try:
        serve(operations, bus, host, port, model_gateway)
    finally:
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


def _install_shutdown(operations) -> None:
    """Stop the engines when we are asked to quit.

    Python does not run its exit handlers on SIGTERM, so a plain `kill` would
    otherwise leave engines running and holding their ports. The next manager
    would then find a stranger answering its health probe. Where systemd owns
    the engines this is a no-op, because there they are meant to survive.
    """
    def shutdown(_signal, _frame):
        operations.host.stop_all()
        sys.exit(0)

    for received in (signal.SIGTERM, signal.SIGINT):
        signal.signal(received, shutdown)


if __name__ == "__main__":
    main()
