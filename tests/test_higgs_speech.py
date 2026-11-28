import base64
import json
import shutil
import socket
import sys
import tempfile
import time
import unittest
from pathlib import Path
from threading import BoundedSemaphore
from unittest.mock import patch

from ai_lab.speech.higgs_backend import HiggsBackend, wait_for_free_port


class FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def read(self):
        return b'RIFFfake-wav'


class HiggsRequestTests(unittest.TestCase):
    def setUp(self):
        self.backend = HiggsBackend.__new__(HiggsBackend)
        self.backend.model_name = 'future-checkpoint'
        self.backend.model_path = Path('/models/future-checkpoint')
        self.backend.worker_port = 8122
        self.backend.slots = BoundedSemaphore(8)

    def test_converts_binary_worker_audio_to_speech_contract(self):
        with patch('urllib.request.urlopen', return_value=FakeResponse()) as call:
            result = self.backend.generate({'model': 'future-checkpoint',
                                            'text': 'Hello', 'speaker': 'default'})
        payload = json.loads(call.call_args.args[0].data)
        self.assertEqual(payload['model'], '/models/future-checkpoint')
        self.assertEqual(payload['input'], 'Hello')
        self.assertEqual(base64.b64decode(result['data'][0]['b64_wav']),
                         b'RIFFfake-wav')

    def test_instruction_is_not_silently_ignored(self):
        with self.assertRaisesRegex(ValueError, 'inline'):
            self.backend.generate({'text': 'Hello', 'instruction': 'calm'})


class WorkerPortTests(unittest.TestCase):
    def test_a_free_port_is_accepted_at_once(self):
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', 0))
            port = probe.getsockname()[1]
        started = time.monotonic()
        wait_for_free_port(port, timeout_s=1)
        self.assertLess(time.monotonic() - started, 0.5)       # no waiting at all

    def test_a_held_port_is_reported_instead_of_silently_moved(self):
        with socket.socket() as holder:
            holder.bind(('127.0.0.1', 0))
            holder.listen()
            with self.assertRaisesRegex(RuntimeError, 'still in use'):
                wait_for_free_port(holder.getsockname()[1], timeout_s=0)


WORKER = r"""import json, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
port = int(sys.argv[sys.argv.index("--port") + 1])
log = sys.argv[0] + ".asked"

class Worker(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200 if self.path == "/health" else 404)
        self.end_headers()

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        reference = body.get("references", [{}])[0].get("audio_path")
        body["reference_bytes"] = open(reference, "rb").read().hex() if reference else None
        open(log, "a").write(json.dumps(body) + "\n")
        audio = b"RIFF" + b"\0" * 40 if "not-wav" not in body["input"] else b"oops"
        self.send_response(200)
        self.send_header("Content-Length", str(len(audio)))
        self.end_headers()
        self.wfile.write(audio)

    def log_message(self, *_):
        pass

HTTPServer(("127.0.0.1", port), Worker).serve_forever()
"""


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class HiggsWorkerTests(unittest.TestCase):
    """The real constructor starts a stand-in SGLang-Omni worker and waits for its /health."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        (self.root / "higgs-audio-v3").mkdir()
        self.worker = self.root / "sgl-omni"
        self.worker.write_text(f"#!{sys.executable}\n{WORKER}")
        self.worker.chmod(0o755)

    def started(self, **changes) -> HiggsBackend:
        made = {"worker_binary": self.worker, "model_path": self.root / "higgs-audio-v3",
                "worker_port": free_port(), "mem_fraction_static": 0.5, **changes}
        backend = HiggsBackend(**made)
        self.addCleanup(backend.close)
        return backend

    def asked(self) -> list[dict]:
        return [json.loads(line) for line in Path(str(self.worker) + ".asked").read_text().splitlines()]

    def test_a_line_is_said_with_its_seed_and_a_cloned_voice_and_the_clip_removed(self):
        backend = self.started()
        clip = b"RIFF" + b"clip" * 10
        answer = backend.generate({"model": "higgs-audio-v3", "text": "Hello", "seed": 3,
                                   "reference_audio": base64.b64encode(clip).decode(),
                                   "reference_text": "Hi"})
        asked, = self.asked()
        self.assertEqual((asked["input"], asked["voice"], asked["seed"]), ("Hello", "default", 3))
        self.assertEqual(bytes.fromhex(asked["reference_bytes"]), clip)
        self.assertEqual(asked["references"][0]["text"], "Hi")
        self.assertFalse(Path(asked["references"][0]["audio_path"]).exists())
        self.assertEqual((answer["model"], answer["mode"]), ("higgs-audio-v3", "higgs"))

    def test_an_answer_that_is_not_a_wav_is_an_error(self):
        backend = self.started()
        with self.assertRaisesRegex(RuntimeError, "did not return WAV"):
            backend.generate({"text": "not-wav please"})

    def test_close_stops_the_worker(self):
        backend = self.started()
        backend.close()
        self.assertIsNotNone(backend.process.poll())

    def test_a_worker_that_exits_during_startup_is_reported(self):
        self.worker.write_text(f"#!{sys.executable}\nimport sys\nsys.exit(3)\n")
        with self.assertRaisesRegex(RuntimeError, "exited during startup"):
            self.started()

    def test_an_absent_worker_or_bad_settings_are_refused_before_starting(self):
        with self.assertRaisesRegex(ValueError, "absent"):
            self.started(worker_binary=self.root / "none")
        for changes in ({"worker_port": 0}, {"mem_fraction_static": 1.0}, {"max_parallel": 65}):
            with self.subTest(changes), self.assertRaisesRegex(ValueError, "settings are invalid"):
                self.started(**changes)


if __name__ == "__main__":
    unittest.main()
