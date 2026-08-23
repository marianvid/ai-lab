"""How much memory is available for models on this machine.

One question, asked in one place, so that the page, the admission check and
(later) the scheduler all get the same answer instead of each working it out
slightly differently.

## Two pools, and why they are not the same

**A dedicated card's memory is used whole.** Nothing else on this machine wants
it — there is no display attached to the card in the container — so the whole
of it is available for models and there is no setting. Somebody running a
desktop off the same card would want to hold some back; that is a reserve like
the one below, and adding it is one field.

**The machine's own memory is shared with everything else** — the browser, the
editor, the operating system — so a part of it is held back. That is the one
setting here: how much to leave for the system. Set as a reserve rather than as
"how much models may use", because a reserve stays right when the machine's
memory changes and the other does not: a container grown from 48 GB to 64 GB
should offer models the extra 16, not sit on it because a number said 40.

**On Apple silicon there is only one pool.** The chip and everything else share
the same memory, so the card reading and the machine reading are two views of
one thing and adding them would count it twice. The reserve applies to the one
pool.

## Why "available" is counted conservatively

Being wrong in the optimistic direction is the expensive one. Refusing a model
that would have fitted costs a sentence on screen; starting one that does not
fit costs the model that was already working. So a pool reports as used
whatever cannot be taken back without somebody noticing, and everything else is
free.
"""

from __future__ import annotations

from dataclasses import dataclass

# What to leave for the machine itself, when nothing says otherwise. Enough for
# an operating system, a browser and an editor to stay comfortable; small
# enough not to waste a third of a small machine.
DEFAULT_RESERVE_MB = 8192

CARD = "card"
MACHINE = "machine"


@dataclass(frozen=True, slots=True)
class Pool:
    """One place a model's memory can go.

    `used_by_models` is what this application's own engines hold — known
    exactly, because it starts them. `used` is everything, including the rest
    of the machine. On a dedicated card the two are nearly the same; on a
    laptop somebody is also working on they are not.

    `available` is what a new model may take: what is free, less the reserve.
    Never negative — a machine already past its reserve has nothing to offer,
    not a negative amount.
    """

    name: str
    kind: str                     # "dedicated" or "unified"
    total_mb: float
    used_mb: float
    used_by_models_mb: float
    reserve_mb: float

    @property
    def free_mb(self) -> float:
        return max(0.0, self.total_mb - self.used_mb)

    @property
    def available_mb(self) -> float:
        return max(0.0, self.free_mb - self.reserve_mb)

    def json(self) -> dict:
        return {"name": self.name, "kind": self.kind,
                "total_mb": round(self.total_mb),
                "used_mb": round(self.used_mb),
                "used_by_models_mb": round(self.used_by_models_mb),
                "reserve_mb": round(self.reserve_mb),
                "free_mb": round(self.free_mb),
                "available_mb": round(self.available_mb)}


@dataclass(frozen=True, slots=True)
class Budget:
    """What this machine can offer a model right now.

    `pools` is one entry on Apple silicon and two on a machine with a dedicated
    card. `for_models_mb` is the total across them, which is the number to put
    on a page — but a model does not get to spread itself freely across both,
    so the pools are what an admission decision has to look at.
    """

    pools: tuple[Pool, ...]
    unified: bool

    @property
    def for_models_mb(self) -> float:
        return sum(pool.available_mb for pool in self.pools)

    @property
    def held_by_models_mb(self) -> float:
        # On unified memory the two readings are two views of one pool, so the
        # card's figure is the whole answer and the pools do not add up.
        return sum(pool.used_by_models_mb for pool in self.pools)

    def pool(self, name: str) -> "Pool | None":
        return next((item for item in self.pools if item.name == name), None)

    def json(self) -> dict:
        return {"pools": [pool.json() for pool in self.pools],
                "unified": self.unified,
                "for_models_mb": round(self.for_models_mb),
                "held_by_models_mb": round(self.held_by_models_mb)}


def of(host, reserve_mb: float = DEFAULT_RESERVE_MB) -> Budget:
    """Read this machine and say what it can offer.

    Never raises and never returns nothing: a machine that cannot be read
    reports pools of zero, which reads as "no answer" rather than as "no room",
    and the caller decides what to do about that.
    """
    try:
        card = host.accelerator()
    except Exception:
        card = None
    try:
        machine_used, machine_total = host.system_memory()
    except Exception:
        machine_used, machine_total = 0.0, 0.0

    unified = bool(card and card.memory_kind == "unified")

    if unified:
        # One pool. The machine reading knows about everything, so it is the
        # one to trust for what is free; the card reading knows what our own
        # engines hold, which the machine reading cannot separate out.
        return Budget(unified=True, pools=(Pool(
            name=MACHINE, kind="unified",
            total_mb=machine_total or (card.memory_total_mb if card else 0.0),
            used_mb=machine_used,
            used_by_models_mb=card.memory_used_mb if card else 0.0,
            reserve_mb=reserve_mb),))

    pools = []
    if card and card.available and card.memory_total_mb:
        # The whole card is for models: nothing else on this machine wants it.
        pools.append(Pool(name=CARD, kind="dedicated",
                          total_mb=card.memory_total_mb,
                          used_mb=card.memory_used_mb,
                          used_by_models_mb=card.memory_used_mb,
                          reserve_mb=0.0))
    if machine_total:
        # Where the part of a model that does not fit on the card goes, and
        # where everything an engine keeps outside the card lives.
        pools.append(Pool(name=MACHINE, kind="dedicated",
                          total_mb=machine_total,
                          used_mb=machine_used,
                          used_by_models_mb=0.0,
                          reserve_mb=reserve_mb))
    return Budget(pools=tuple(pools), unified=False)
