"""The reservation routes, the front door's check and the Load buttons' refusal."""

import tempfile
import unittest
from http import HTTPStatus
from pathlib import Path
from typing import Any, cast

from ai_lab.api.router import Router
from ai_lab.api.routes import reservations as reservation_routes
from ai_lab.api.routes import runtime as runtime_routes
from ai_lab.api.routes.gateway import RESERVATION_HEADER, _forwarder
from ai_lab.api.server import status_for
from ai_lab.config import ConfigStore
from ai_lab.lanes import SideLanes
from ai_lab.reservations.book import ReservationBook
from ai_lab.reservations.errors import CardReservedError, ReservationEndedError, ReservationRequiredError
from ai_lab.reservations.settings import ReservationPolicy


class Operations:
    def __init__(self):
        path = Path(tempfile.mkdtemp()) / "config.json"
        path.write_text('{"models_root": "/tmp"}')
        self.store = ConfigStore(path)
        self.loaded = []

    def load(self, instance_id):
        self.loaded.append(instance_id)
        return type("Done", (), {"json": lambda self: {"ok": True}})()


class Lane:
    def __init__(self, names):
        self.names = names

    def serves(self, name):
        return name in self.names

    def complete(self, path, payload):
        return {"answered": payload["model"]}

    embed = complete


class Gateway:
    def acquire(self, wanted, **_):
        raise LookupError(f"card asked for {wanted}")


def call(router, method, path, **kwargs):
    handler, captured = router.match(method, path)
    return handler(**captured, **kwargs)


class RouteTests(unittest.TestCase):
    def setUp(self):
        self.book = ReservationBook(ReservationPolicy(unreserved_wait_s=0))
        self.operations = Operations()
        self.router = Router()
        reservation_routes.register(self.router, self.operations, self.book)
        runtime_routes.register(self.router, self.operations, None, self.book)

    def test_ask_beat_extend_release(self):
        made = call(self.router, "POST", "/api/reservations",
                    body={"team": "ingest", "instance": "ct109", "priority": 4})
        rid = made["id"]
        self.assertEqual(call(self.router, "GET", f"/api/reservations/{rid}")["state"], "granted")
        self.assertEqual(call(self.router, "POST", f"/api/reservations/{rid}/heartbeat")["id"], rid)
        self.assertEqual(call(self.router, "POST", f"/api/reservations/{rid}/extend",
                              body={"seconds": 60})["max_s"], 1860)
        self.assertEqual(call(self.router, "GET", "/api/reservations")["holder"]["id"], rid)
        self.assertEqual(call(self.router, "DELETE", f"/api/reservations/{rid}")["state"], "ended")

    def test_the_load_button_waits_for_the_reservation(self):
        made = call(self.router, "POST", "/api/reservations", body={"team": "ingest"})
        with self.assertRaisesRegex(CardReservedError, "cancel the reservation"):
            call(self.router, "POST", "/api/instances/qwen/load", body={"force": True})
        call(self.router, "POST", f"/api/reservations/{made['id']}/cancel")
        call(self.router, "POST", "/api/instances/qwen/load", body={})
        self.assertEqual(self.operations.loaded, ["qwen"])

    def test_settings_are_kept_and_used(self):
        answer = call(self.router, "PATCH", "/api/reservations/settings",
                      body={"ceiling_s": 7200, "required": True})
        self.assertEqual(answer["settings"]["ceiling_s"], 7200)
        self.assertEqual(self.operations.store.load().reservations["ceiling_s"], 7200)
        self.assertTrue(self.book.policy.required)
        with self.assertRaises(ValueError):
            call(self.router, "PATCH", "/api/reservations/settings", body={"aging_s": -1})

    def test_statuses(self):
        self.assertEqual(status_for(CardReservedError("x")), HTTPStatus.CONFLICT)
        self.assertEqual(status_for(ReservationEndedError("x")), HTTPStatus.CONFLICT)
        self.assertEqual(status_for(ReservationRequiredError("x")), HTTPStatus.PRECONDITION_REQUIRED)


class FrontDoorTests(unittest.TestCase):
    def setUp(self):
        self.book = ReservationBook(ReservationPolicy(unreserved_wait_s=0))
        self.lanes = SideLanes(providers=cast(Any, Lane({"claude/sonnet"})),
                               cpu=cast(Any, Lane({"bge-m3"})))
        self.handle = _forwarder(cast(Any, Gateway()), "/v1/chat/completions",
                                 self.lanes, self.book)

    def test_only_the_holder_reaches_this_machine_s_models(self):
        rid = self.book.request({"team": "ingest"})["id"]
        mine = {RESERVATION_HEADER: rid}
        self.assertEqual(self.handle(body={"model": "bge-m3"}, headers=mine),
                         {"answered": "bge-m3"})
        with self.assertRaisesRegex(LookupError, "card asked"):
            self.handle(body={"model": "qwen"}, headers=mine)
        for model in ("bge-m3", "qwen"):
            with self.subTest(model=model), self.assertRaises(CardReservedError):
                self.handle(body={"model": model}, headers={})

    def test_cloud_models_are_never_held_back(self):
        self.book.request({"team": "ingest"})
        self.assertEqual(self.handle(body={"model": "claude/sonnet"}, headers={}),
                         {"answered": "claude/sonnet"})


if __name__ == "__main__":
    unittest.main()
