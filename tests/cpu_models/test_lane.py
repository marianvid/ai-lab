"""The processor lane: its own queue, start on demand, idle stop, and the server's answer."""

import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast

from ai_lab.cpu_models import EMBEDDINGS_PATH, CpuLane, CpuModelsConfig
from ai_lab.cpu_models.lane import LaneTools, _post

RAW = {"binary": "/bin/llama-server",
       "models": {"bge-m3": {"path": "/m/bge.gguf", "port": 8390, "parallel": 2}}}
REQUEST = {"model": "bge-m3", "input": ["a text"]}


SHARED = {"binary": "/bin/llama-server",
          "models": {"bge-m3": {"path": "/m/bge.gguf", "port": 8390, "parallel": 2,
                                "card": {"port": 8391, "needs_mb": 1300}}}}


class FakeServer:
    def __init__(self, binary, model, log_dir, device):
        self.model = model
        self.device = device
        self.port = model.card.port if device == "card" else model.port
        self.running = False
        self.last_used = 0.0
        self.starts = self.stops = 0

    def ensure_started(self):
        if not self.running:
            self.starts += 1
        self.running = True

    def touch(self):
        self.last_used = 5.0

    def stop(self):
        self.stops += 1
        self.running = False

    def stop_if_idle(self, busy):
        if busy or not self.running:
            return False
        self.stop()
        return True


class CpuLaneTests(unittest.TestCase):
    def setUp(self):
        self.posted = []
        self.free = None
        self.lane = self.make(RAW)

    def make(self, raw, poster=None):
        tools = LaneTools(poster=poster or self.post, server_factory=cast(Any, FakeServer),
                          card_free_mb=lambda: self.free, clock=lambda: 10.0)
        return CpuLane(CpuModelsConfig.from_mapping(raw), Path(tempfile.mkdtemp()), tools)

    def post(self, port, payload):
        self.posted.append((port, payload))
        return {"object": "list", "data": [{"index": 0, "embedding": [0.1]}], "model": "x"}

    def test_a_request_starts_the_model_and_gets_its_answer(self):
        answer = self.lane.embed(EMBEDDINGS_PATH, REQUEST)
        self.assertEqual(answer["model"], "bge-m3")
        self.assertEqual(answer["ai_lab"], {"device": "cpu"})
        self.assertEqual(self.posted, [(8390, REQUEST)])
        stats = self.lane.stats()["bge-m3"]
        self.assertEqual((stats["running"], stats["requests"], stats["in_flight"]), (True, 1, 0))
        self.assertEqual((stats["places"], stats["idle_s"]), (2, 0.0))

    def test_only_embeddings(self):
        with self.assertRaisesRegex(ValueError, "only /v1/embeddings"):
            self.lane.embed("/v1/chat/completions", REQUEST)

    def test_a_failure_is_counted_and_raised(self):
        def broken(_port, _payload):
            raise ValueError("input is too long")
        self.lane = self.make(RAW, broken)
        with self.assertRaisesRegex(ValueError, "too long"):
            self.lane.embed(EMBEDDINGS_PATH, REQUEST)
        self.assertEqual(self.lane.stats()["bge-m3"]["failures"], 1)

    def test_at_most_parallel_requests_reach_the_server(self):
        inside, peak, release = [0], [0], threading.Event()

        def slow(_port, _payload):
            inside[0] += 1
            peak[0] = max(peak[0], inside[0])
            release.wait(2)
            inside[0] -= 1
            return {"data": []}
        self.lane = self.make(RAW, slow)
        threads = [threading.Thread(target=self.lane.embed, args=(EMBEDDINGS_PATH, REQUEST))
                   for _ in range(5)]
        for thread in threads:
            thread.start()
        while self.lane.stats()["bge-m3"]["waiting"] < 3:
            pass
        release.set()
        for thread in threads:
            thread.join()
        self.assertEqual(peak[0], 2)

    def test_start_stop_by_hand_and_idle_reaping(self):
        self.assertTrue(self.lane.start("bge-m3")["running"])
        self.assertEqual(self.lane.reap_idle(), ["bge-m3/cpu"])
        self.lane.start("bge-m3")
        self.assertFalse(self.lane.stop("bge-m3")["running"])
        with self.assertRaises(KeyError):
            self.lane.start("nothing")
        self.lane.stop_all()

    def test_a_busy_model_is_not_stopped_by_hand(self):
        self.lane._queues["bge-m3"].in_flight = 1
        with self.assertRaisesRegex(ValueError, "answering requests"):
            self.lane.stop("bge-m3")

    def test_catalogue_names_the_processor_lane(self):
        row = self.lane.catalogue()[0]
        self.assertEqual((row["id"], row["ai_lab"]["lane"], row["ai_lab"]["shapes"]),
                         ("bge-m3", "cpu", [EMBEDDINGS_PATH]))
        self.assertTrue(self.lane.serves("bge-m3"))
        self.assertFalse(self.lane.serves("qwen36-nvfp4"))

    def test_built_from_configuration(self):
        lane = CpuLane.from_config(RAW, Path(tempfile.mkdtemp()))
        self.assertEqual(list(lane.placements), ["bge-m3"])
        self.assertEqual(lane.reclaimable_mb(), 0.0)

    def test_with_room_on_the_card_it_answers_there_and_gives_it_back(self):
        self.free = 4000.0
        lane = self.make(SHARED)
        answer = lane.embed(EMBEDDINGS_PATH, REQUEST)
        self.assertEqual((answer["ai_lab"]["device"], self.posted[-1][0]), ("card", 8391))
        stats = lane.stats()["bge-m3"]
        self.assertEqual((stats["device"], stats["card"], stats["card_port"],
                          stats["card_in_flight"]), ("card", True, 8391, 0))
        self.assertEqual(lane.reclaimable_mb(), 1300.0)
        lane.make_way()
        self.assertEqual(lane.reclaimable_mb(), 0.0)
        self.assertEqual(lane.embed(EMBEDDINGS_PATH, REQUEST)["ai_lab"]["device"], "cpu")
        lane.settled()          # back to the card, in the background
        for _ in range(200):
            if lane.catalogue()[0]["ai_lab"]["device"] == "card":
                break
            threading.Event().wait(0.01)
        self.assertEqual(lane.catalogue()[0]["ai_lab"]["device"], "card")


class Embeddings(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if body.get("input") == ["too long"]:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b'{"error": "input is too long"}')
            return
        data = json.dumps({"data": [{"index": 0, "embedding": [1.0]}]}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *_):
        pass


class PostTests(unittest.TestCase):
    def setUp(self):
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Embeddings)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.shutdown)
        self.port = self.server.server_address[1]

    def test_the_answer_and_the_refusal_are_the_servers_own(self):
        self.assertEqual(_post(self.port, {"input": ["x"]})["data"][0]["embedding"], [1.0])
        with self.assertRaisesRegex(ValueError, "too long"):
            _post(self.port, {"input": ["too long"]})


if __name__ == "__main__":
    unittest.main()
