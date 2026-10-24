"""The status command: `ai-lab-gpu-health status` (or `summary`, `alerts`, `report`).

It asks the running AI-Lab (on this machine, at the default port 8090, unless
`--url` says otherwise) rather than opening the database itself, so it never
touches the file the monitor writes and shows the same as the API.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request

from ...config.store import Config
from ...network import local_url

# Where the manager answers when its configuration keeps the default port.
DEFAULT_URL = local_url(Config().port)
NOW = ("temp_gpu", "power_w", "fan_pct", "util_pct", "pcie_gen", "pcie_width", "model")


def fetch(url: str) -> dict:
    """GET a JSON answer."""
    with urllib.request.urlopen(url, timeout=30) as answer:  # noqa: S310 - our own address
        return json.loads(answer.read().decode())


def when(stamp: float | None) -> str:
    """A moment as local time, or "never"."""
    return "never" if not stamp else time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stamp))


def describe(state: dict) -> str:
    """The status as plain lines."""
    if not state.get("available"):
        return "GPU health: not available here. " + state.get("reason", "")
    return "\n".join([*_now(state), *_reference(state["analysis"]), *_problems(state)])


def _now(state: dict) -> list[str]:
    current = state.get("current") or {}
    return [f"GPU health monitor: {'running' if state['running'] else 'STOPPED'}; "
            f"last reading {when(state.get('last_ok'))}",
            "Now: " + ", ".join(f"{name} {current.get(name)}" for name in NOW if name in current),
            "Not available on this card: " + (", ".join(state["not_available"]) or "none")]


def _reference(analysis: dict) -> list[str]:
    idle = analysis.get("idle_today")
    lines = ["Resting temperature today: " +
             (f"{idle['temp']:.1f} °C (from {idle['from_day']})" if idle else "no reference")]
    lines += [f"Reference {scope}: R {base['r']:.3f} °C/W from {base['windows']} windows "
              f"({base['period_start']} – {base['period_end']})"
              for scope, base in analysis["baselines"].items()]
    if not analysis["baselines"]:
        lines.append("Reference: not formed yet (first 21 days, at least 10 windows)")
    return lines


def _problems(state: dict) -> list[str]:
    lines = [f"Active alarms: {len(state['alarms'])}"]
    lines += [f"  {item['level'].upper()}: {item['message']}" for item in state["alarms"]]
    helper = state["helper"]
    lines.append("Host helper: " + (f"last heartbeat {when(helper['last_heartbeat'])}"
                                    if helper["installed"] else "not installed"))
    lines += [f"Step {name} failed {item['count']} times, last: {item.get('error')}"
              for name, item in state["failures"].items()]
    lines.append("Rows: " + ", ".join(f"{name} {count}" for name, count in state["counts"].items()))
    return lines


def main(argv: list[str] | None = None) -> int:
    """`status` or `report [--days N]`; 0 when the answer came, 1 when it did not."""
    parser = argparse.ArgumentParser(prog="python -m ai_lab.gpu_health",
                                     description="The GPU health monitor's status")
    parser.add_argument("what", choices=("status", "summary", "alerts", "report"), nargs="?",
                        default="status")
    parser.add_argument("--url", default=DEFAULT_URL, help="where AI-Lab answers")
    parser.add_argument("--days", type=int, default=7, help="the report's period")
    parser.add_argument("--after", type=int, default=0,
                        help="alerts: only those after this id")
    parser.add_argument("--json", action="store_true", help="print the raw answer")
    arguments = parser.parse_args(argv)
    path = PATHS[arguments.what].format(days=arguments.days, after=arguments.after)
    try:
        answer = fetch(arguments.url.rstrip("/") + path)
    except OSError as error:
        print(f"AI-Lab did not answer at {arguments.url}: {error}", file=sys.stderr)
        return 1
    if arguments.json:
        print(json.dumps(answer, indent=2))
    else:
        print(SHOW[arguments.what](answer))
    return 0


def describe_summary(answer: dict) -> str:
    """The short state as one line per alarm."""
    if not answer.get("available"):
        return "GPU health: not available here. " + answer.get("reason", "")
    lines = [f"GPU health: {answer['health'].upper()} · monitor {answer['monitor']} · "
             f"last reading {when(answer.get('last_reading'))} · last alert id "
             f"{answer['last_alert_id']}"]
    lines += [f"  {item['level'].upper()}: {item['message']}" for item in answer["alarms"]]
    return "\n".join(lines)


def describe_alerts(answer: dict) -> str:
    """The messages, one per line, oldest first, with their id."""
    lines = [f"{item['id']} {when(item['ts'])} {item['level'].upper()} [{item['kind']}] "
             f"{item['message']}" for item in answer["alerts"]]
    lines.append(f"last id {answer['last_id']}" + (" (more: ask again with --after)"
                                                   if answer["more"] else ""))
    return "\n".join(lines)


PATHS = {"status": "/api/gpu-health", "summary": "/api/gpu-health/summary",
         "alerts": "/api/gpu-health/alerts?after={after}",
         "report": "/api/gpu-health/report?days={days}"}
SHOW = {"status": describe, "summary": describe_summary, "alerts": describe_alerts,
        "report": lambda answer: answer.get("text", "")}
