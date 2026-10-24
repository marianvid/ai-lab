"""A reservation checked again when the call is served, not only at the door (ADR 0005 §10a)."""

import threading
import time
import unittest

from ai_lab.api.routes.gateway import RESERVATION_HEADER, _forwarder
from ai_lab.reservations.book import ReservationBook
from ai_lab.reservations.errors import (
    CardReservedError,
    ReservationEndedError,
    ReservationNotGrantedError,
)
from ai_lab.reservations.settings import ReservationPolicy
from ai_lab.gateway.admission.scheduler import AdmissionWithdrawnError, Scheduler


class Card:
    """A card that records what was put on it."""

    def __init__(self):
        self.switches = []

    def switch(self, shape, victims=()):
        self.switches.append(shape)

    @staticmethod
    def places(_shape):
        return 1


def entered(scheduler, shape, still_wanted=None):
    """Start a request in its own thread; the box gets None or the error it raised."""
    box: dict[str, BaseException | None] = {}

    def run():
        try:
            scheduler.enter(shape, still_wanted=still_wanted)
            box["error"] = None
        except BaseException as error:  # noqa: BLE001 - recorded for the test
            box["error"] = error
    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread, box


def wait_until(condition, limit_s=2.0):
    deadline = time.monotonic() + limit_s
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError("condition not reached")
        time.sleep(0.005)


class PermitsTests(unittest.TestCase):
    def setUp(self):
        self.book = ReservationBook(ReservationPolicy(unreserved_wait_s=0))

    def test_without_a_reservation_only_while_the_card_is_free(self):
        self.book.permits(None)
        self.book.request({"team": "ingest", "priority": 4})
        with self.assertRaises(CardReservedError):
            self.book.permits(None)

    def test_with_a_reservation_only_while_it_holds_the_card(self):
        holder = self.book.request({"team": "ingest", "priority": 4})
        waiting = self.book.request({"team": "music", "priority": 2})
        self.book.permits(holder["id"])
        with self.assertRaises(ReservationNotGrantedError):
            self.book.permits(waiting["id"])
        self.book.release(holder["id"])
        with self.assertRaises(ReservationEndedError):
            self.book.permits(holder["id"])
        self.book.permits(waiting["id"])


class SchedulerTests(unittest.TestCase):
    """The scheduler drops a withdrawn request before anything is loaded for it."""

    def setUp(self):
        self.card = Card()
        self.scheduler = Scheduler(self.card.switch, self.card.places)
        self.scheduler.enter("a")                       # a long call holds "a"

    def test_a_withdrawn_request_fails_with_its_reason_and_loads_nothing(self):
        refused = CardReservedError("the card is reserved by ingest")

        def withdrawn():
            raise AdmissionWithdrawnError(refused)
        thread, box = entered(self.scheduler, "b", withdrawn)
        wait_until(lambda: self.scheduler.state()["waiting"])
        self.scheduler.leave("a")
        thread.join(2)
        self.assertIsInstance(box["error"], AdmissionWithdrawnError)
        self.assertIs(box["error"].reason, refused)
        self.assertEqual(self.card.switches, ["a"])     # nothing loaded for "b"

    def test_a_holder_s_swap_waits_only_for_the_call_in_flight(self):
        thread, box = entered(self.scheduler, "b", lambda: True)
        wait_until(lambda: self.scheduler.state()["waiting"])
        self.assertEqual(self.card.switches, ["a"])     # "a" still answering
        self.scheduler.leave("a")
        thread.join(2)
        self.assertIsNone(box["error"])
        self.assertEqual(self.card.switches, ["a", "b"])


class Lease:
    def __init__(self):
        self.released = False

    def release(self):
        self.released = True


class QueuedGateway:
    """Stands in for the card: `before_head()` runs while the call waits in its queue."""

    def __init__(self, before_head=None):
        self.before_head = before_head
        self.acquired = 0
        self.asked = 0

    def acquire(self, wanted, still_wanted=None, **_):
        self.asked += 1
        if self.before_head is not None:
            self.before_head()
            self.before_head = None
        if not still_wanted():
            raise AssertionError("the client is still there")
        self.acquired += 1
        raise LookupError("served")                     # stops before forwarding


class ForwarderTests(unittest.TestCase):
    def forward(self, gateway, book, reservation=None):
        headers = {RESERVATION_HEADER: reservation} if reservation else {}
        handle = _forwarder(gateway, "/v1/chat/completions", book=book)
        return handle(body={"model": "qwen", "messages": []}, alive=lambda: True,
                      headers=headers)

    def test_an_unreserved_call_overtaken_by_a_reservation_waits_at_the_door(self):
        book = ReservationBook(ReservationPolicy(unreserved_wait_s=0))
        gateway = QueuedGateway(lambda: book.request({"team": "ingest", "priority": 4}))
        with self.assertRaisesRegex(CardReservedError, "reserved by ingest"):
            self.forward(gateway, book)
        self.assertEqual((gateway.asked, gateway.acquired), (1, 0))

    def test_it_is_served_once_the_reservation_is_released(self):
        book = ReservationBook(ReservationPolicy(unreserved_wait_s=5))
        holder = {}

        def reserve_then_release_soon():
            holder.update(book.request({"team": "ingest", "priority": 4}))
            threading.Timer(0.2, book.release, args=(holder["id"],)).start()
        gateway = QueuedGateway(reserve_then_release_soon)
        with self.assertRaisesRegex(LookupError, "served"):
            self.forward(gateway, book)
        self.assertEqual((gateway.asked, gateway.acquired), (2, 1))

    def test_a_holder_whose_reservation_ended_while_queued_gets_409(self):
        book = ReservationBook(ReservationPolicy(unreserved_wait_s=0))
        made = book.request({"team": "ingest", "priority": 4})
        gateway = QueuedGateway(lambda: book.release(made["id"]))
        with self.assertRaisesRegex(ReservationEndedError, "has ended"):
            self.forward(gateway, book, made["id"])
        self.assertEqual(gateway.acquired, 0)

    def test_the_holder_is_served(self):
        book = ReservationBook(ReservationPolicy(unreserved_wait_s=0))
        made = book.request({"team": "ingest", "priority": 4})
        gateway = QueuedGateway()
        with self.assertRaisesRegex(LookupError, "served"):
            self.forward(gateway, book, made["id"])
        self.assertEqual(gateway.acquired, 1)
