"""Where one model answers: on the card when there is room, else on the processor.

The rule (decided 2026-10-15, `docs/decisions/0004-embeddings-borrow-the-card.md`):

- A model that may use the card goes there when it is asked for and the card has
  room for it (`CardUse.room_mb` free). Otherwise it answers on the processor.
- The card's own models come first. Before any of them is loaded — through the
  gateway, the Load button or a restart — `make_way` moves this model off the
  card: new requests go to the processor, those already on the card finish
  there, then its card server stops.
- After that load, `settled` brings it back to the card if there is still room
  and it was used within its idle limit.
- While it answers on the processor, every request checks whether the card has
  room again — at most once every `RECHECK_S` after a "no", because reading the
  card costs a call to the driver.

Both servers may run at once for a while; the processor's stops when idle.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

from .server import CARD, CPU, CpuServer

# How long a "no room" answer is trusted before the card is read again.
RECHECK_S = 30.0
# How long requests already on the card may take to finish before the card
# server is stopped anyway. A call of 300 sentences takes half a second there.
DRAIN_LIMIT_S = 60.0
# A safety net: if a card load never reports back, the card is offered again
# after this long rather than never.
HOLD_LIMIT_S = 900.0
POLL_S = 0.05

CardRoom = Callable[[], "float | None"]


class Placement:
    """One model's two servers, and which of them takes the next request."""

    def __init__(self, cpu: CpuServer, card: CpuServer | None, card_free_mb: CardRoom,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self.cpu = cpu
        self.card = card
        self.model = cpu.model
        self._card_free_mb = card_free_mb
        self._clock = clock
        self._sleep = sleep
        self._state = threading.Lock()       # the counters and flags below
        self._card_turn = threading.Lock()   # starting or stopping the card server
        self._on_card = 0
        self._held_until = 0.0
        self._no_room_until = 0.0
        self.last_used = 0.0

    # -- one request ---------------------------------------------------------

    def take(self) -> CpuServer:
        """The server for one request, started if need be; hand it back with `give_back`."""
        self.last_used = self._clock()
        if self.card is not None and self._to_card(with_request=True):
            return self.card
        self.cpu.ensure_started()
        return self.cpu

    def give_back(self, server: CpuServer) -> None:
        """That request is finished."""
        server.touch()
        if server is self.card:
            with self._state:
                self._on_card -= 1

    # -- the card's own models come first -------------------------------------

    def make_way(self) -> None:
        """Leave the card before a card model is loaded."""
        if self.card is None:
            return
        with self._state:
            self._held_until = self._clock() + HOLD_LIMIT_S
        with self._card_turn:
            deadline = self._clock() + DRAIN_LIMIT_S
            while self._card_busy() and self._clock() < deadline:
                self._sleep(POLL_S)
            self.card.stop()

    def settled(self) -> None:
        """A card load is over: come back to the card if there is room and work."""
        if self.card is None:
            return
        with self._state:
            self._held_until = 0.0
            self._no_room_until = 0.0
        if self._in_demand():
            threading.Thread(target=self._to_card, daemon=True,
                             name=f"{self.model.name}-to-card").start()

    def reclaimable_mb(self) -> float:
        """Card memory this model holds and gives back whenever a card model needs it."""
        if self.card is not None and self.model.card and self.card.running:
            return float(self.model.card.needs_mb)
        return 0.0

    # -- by hand, and when idle ----------------------------------------------

    def start(self) -> None:
        """Start it now, on the card if there is room."""
        if self.card is None or not self._to_card():
            self.cpu.ensure_started()

    def stop(self) -> None:
        """Stop both servers."""
        if self.card is not None:
            with self._card_turn:
                self.card.stop()
        self.cpu.stop()

    def stop_if_idle(self, busy: bool) -> list[str]:
        """Stop whichever server nothing used for the idle limit; the devices stopped."""
        stopped = [CPU] if self.cpu.stop_if_idle(busy) else []
        if self.card is not None and self.card.stop_if_idle(busy or self._card_busy()):
            stopped.append(CARD)
        return stopped

    def device(self) -> str:
        """Where the next request would go now: card, cpu, or stopped."""
        if self.card is not None and self.card.running and not self._held():
            return CARD
        return CPU if self.cpu.running else "stopped"

    @property
    def running(self) -> bool:
        """Whether either server is up."""
        return self.cpu.running or bool(self.card and self.card.running)

    def on_card(self) -> int:
        """Requests answering on the card right now."""
        with self._state:
            return self._on_card

    # -- inside ----------------------------------------------------------------

    def _to_card(self, with_request: bool = False) -> bool:
        """Put the model on the card if allowed and there is room; take a place there."""
        card = self.card
        if card is None:
            return False
        with self._state:
            if self._held() or (not card.running and self._clock() < self._no_room_until):
                return False
        with self._card_turn:
            # Checked again here: a card load may have begun while this
            # request waited for its turn.
            if self._held() or (not card.running and not self._start_on_card(card)):
                return False
            with self._state:
                if with_request:
                    self._on_card += 1
                return True

    def _start_on_card(self, card: CpuServer) -> bool:
        free = self._card_free_mb()
        if free is None or not self.model.card or free < self.model.card.room_mb:
            self._no_room(card)
            return False
        try:
            card.ensure_started()
        except Exception:
            # The card refused it after all — another process took the room
            # between the reading and the start. The processor answers instead.
            self._no_room(card)
            return False
        return True

    def _no_room(self, card: CpuServer) -> None:
        card.stop()
        with self._state:
            self._no_room_until = self._clock() + RECHECK_S

    def _held(self) -> bool:
        return self._clock() < self._held_until

    def _card_busy(self) -> bool:
        with self._state:
            return self._on_card > 0

    def _in_demand(self) -> bool:
        if not self.last_used:
            return False
        limit = self.model.idle_stop_s
        return not limit or self._clock() - self.last_used < limit
