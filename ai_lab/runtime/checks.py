"""The checks made before an engine is started, so a failure says why at once.

A base class of `Runtime` (a mixin): let an engine that can split a model
between card and system memory do so when the weights will not fit, refuse a
model larger than the card's free memory, and refuse to start when something
this manager did not start already answers on the port.
"""

from __future__ import annotations

from ..config.store import Instance
from ..engines.base import Engine
from ..hosts.base import Host
from ..types import ModelSet

MIB = 1024 * 1024


class StartChecks:
    """Checks before a start (see `Runtime`, which owns `host`)."""

    host: Host

    def _split_if_it_will_not_fit(self, instance: Instance, model: ModelSet,
                                  engine: Engine) -> dict:
        """Let an engine that can split do so, rather than refusing.

        llama.cpp measures the card at startup and puts on as many layers as
        fit, leaving the rest in system memory. It knows something this cannot:
        the weights are only the floor, and the cache on top of them varies by
        architecture — measured at 32k on the container, the gap between file
        size and card usage ran from -476 MiB to +6,663 across four models. So
        when the weights alone will not fit, the honest move is to hand the
        decision to whoever can measure, not to guess a layer count here.

        vLLM has no such setting: it either fits or it does not, and it says so
        itself in a sentence naming the largest context that would have fitted.
        Nothing is changed for it.

        Only the default is relaxed. `gpu_layers = -1` means "all on the card"
        and is what an entry has when nobody chose; a layer count, or automatic
        already, is somebody's decision and is left exactly as it is.

        The entry's own settings are not touched. What changes is how *this*
        start is made, and the row says so — the same way it does for a request
        that asked for a bigger context than the entry is configured with.
        """
        params = dict(instance.params)
        splitting = getattr(engine, "split_setting", None)
        if not splitting:
            return params
        name, only_when, value = splitting
        if params.get(name) != only_when:
            # A layer count, or automatic already: somebody chose that.
            return params
        snapshot = self.host.accelerator()
        if snapshot.memory_kind != "dedicated" or not snapshot.memory_total_mb:
            return params
        free_mb = snapshot.memory_total_mb - snapshot.memory_used_mb
        if model.size_bytes / MIB <= free_mb:
            return params                       # the weights fit; leave it be
        params[name] = value
        return params

    def _refuse_if_it_cannot_fit(self, instance: Instance, model: ModelSet,
                                 plan) -> None:
        """Refuse a model that is larger than the memory left on the card.

        Only a necessary condition, not a sufficient one: the weights are the
        floor, and the context cache sits on top of them, so a model that
        passes this check can still turn out not to fit. But when the weights
        alone exceed what is free, there is no need to start anything to find
        out — and the alternative is a confusing crash a few seconds later.

        A plan that deliberately leaves part of the model in system memory is
        exempt. There the weights are not all meant to be on the card, so
        comparing the whole file against free memory answers a question nobody
        asked. It is slow, and it is a choice — the setting that turns it on
        says how slow.
        """
        if getattr(plan, "splits_across_cpu", False):
            return
        snapshot = self.host.accelerator()
        if snapshot.memory_kind != "dedicated" or not snapshot.memory_total_mb:
            return
        free_mb = snapshot.memory_total_mb - snapshot.memory_used_mb
        needed_mb = model.size_bytes / MIB
        if needed_mb > free_mb:
            raise RuntimeError(
                f"{model.name} needs about {needed_mb / 1024:.1f} GB but only "
                f"{free_mb / 1024:.1f} GB is free on the card. Unload another "
                f"model, choose a smaller one, or set how many layers go on the "
                f"card and leave the rest in system memory.")

    def _refuse_stranger(self, instance: Instance, engine: Engine) -> None:
        """Refuse to start when something else already holds the port.

        Otherwise the readiness probe answers yes — from a process this
        manager did not start — and a load that never happened is reported as
        a success. The likely culprit is an engine left behind by a previous
        manager, or a second AI-Lab on the same machine.
        """
        if self.host.status(instance.id).running:
            return
        if engine.ready(instance.port):
            raise RuntimeError(
                f"Port {instance.port} is already answering, but this manager did "
                f"not start it. Stop the other process before loading {instance.id}.")
