"""Protect manual load and unload actions while agent requests are active."""

from __future__ import annotations

from .gateway_errors import CardBusy


class GatewayControl:
    def __init__(self, scheduler) -> None:
        self.scheduler = scheduler

    def busy(self) -> dict | None:
        """What is on the card and what it is doing, or None if it is idle.

        Idle means nothing running and nobody waiting. A queue with nothing in
        flight still counts as busy: a switch is about to happen, and stopping
        a model in that moment is as disruptive as stopping one mid-answer.
        """
        state = self.scheduler.state()
        if not state["in_flight"] and not state["waiting"] and not state["switching"]:
            return None
        # Which one to name, when several are loaded: the busiest, because
        # that is the one somebody stopping this would most regret. With one
        # loaded it is the only answer there is.
        busiest = max(state["loaded"], key=lambda item: item["in_flight"],
                      default=None)
        return {
            "instance_id": busiest["shape"].instance_id if busiest else "",
            "answering": bool(state["in_flight"]),
            "in_flight": state["in_flight"],
            "places": busiest["places"] if busiest else 0,
            "waiting": len(state["waiting"]),
            "switching": state["switching"],
        }

    def guard(self, action: str, instance_id: str) -> None:
        """Refuse an action that would interrupt work in progress.

        Requests here are safe from each other: they queue. This is for the
        other way in — the Load and Unload buttons reach the engines directly
        and know nothing about leases or queues. Without this, pressing Unload
        during a long answer kills it mid sentence, and the agent sees a
        connection that simply stopped.

        It names what it found, because "busy" is not enough to decide with:
        one answer being written is a different thing from forty requests
        waiting for a model to load.
        """
        holder = self.busy()
        if holder is None:
            return
        who = holder["instance_id"] or "a model"
        parts = []
        if holder["in_flight"]:
            answers = holder["in_flight"]
            parts.append(f"{who} is answering "
                         f"{answers} request{'' if answers == 1 else 's'}")
        elif holder["switching"]:
            parts.append(f"{who} is being loaded")
        if holder["waiting"]:
            waiting = holder["waiting"]
            parts.append(f"{waiting} more {'is' if waiting == 1 else 'are'} waiting")
        raise CardBusy(
            f"{' and '.join(parts) or 'The card is in use'}. Going ahead with "
            f"'{action}' on {instance_id} cuts all of that off.", holder)
