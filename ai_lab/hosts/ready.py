"""Whether the manager came up after a deploy, and stayed up: `python -m ai_lab.hosts.ready`.

`systemctl is-active` says "active" for a simple unit as soon as its process has
started, so a manager that dies while it builds its catalog still passes it
(the reviewer, on audit m14). This asks the manager itself: `GET /api/session`
must answer within `--within` seconds, and answer again `--settle` seconds
later. The port is read from the configuration given (`--config`), never
written here.

Exit 0 when it answered both times; 1 otherwise, with one line saying why.
"""

from __future__ import annotations

import argparse
import json
import sys
from http import HTTPStatus
import time
import urllib.error
import urllib.request
from pathlib import Path

from ..network import LOOPBACK



def answers(port: int) -> bool:
    """Whether the manager on `port` answers its session question with 200."""
    try:
        with urllib.request.urlopen(f"http://{LOOPBACK}:{port}/api/session", timeout=3) as reply:
            return reply.status == HTTPStatus.OK
    except (OSError, urllib.error.URLError, ValueError):
        return False


def wait(port: int, within: float, settle: float, pause: float = 1.0) -> str:
    """"" when the manager answered, then answered again after `settle`; else why not."""
    deadline = time.monotonic() + within
    while not answers(port):
        if time.monotonic() >= deadline:
            return f"the manager did not answer on port {port} within {within:g} s"
        time.sleep(pause)
    time.sleep(settle)
    if not answers(port):
        return f"the manager answered, then stopped answering within {settle:g} s"
    return ""


def main(argv: list[str] | None = None) -> int:
    """Read the port from the configuration, wait, and say what happened."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="the configuration the manager runs with")
    parser.add_argument("--within", type=float, default=60.0)
    parser.add_argument("--settle", type=float, default=10.0)
    options = parser.parse_args(argv)
    # Read as the file says it, not through `config` (a layer above hosts):
    # the port is the one thing this needs.
    port = json.loads(Path(options.config).read_text()).get("port")
    if not isinstance(port, int):
        print(f"no port in {options.config}", file=sys.stderr)
        return 1
    why = wait(port, options.within, options.settle)
    print(why or f"the manager answers on port {port}", file=sys.stderr if why else sys.stdout)
    return 1 if why else 0


if __name__ == "__main__":
    raise SystemExit(main())
