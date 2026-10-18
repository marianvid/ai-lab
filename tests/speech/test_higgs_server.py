"""The Linux Higgs host: its speech handler and the playground beside it."""

import unittest
from unittest import mock

from ai_lab.media.http_host import GenerationHost
from ai_lab.speech import higgs_server
from ai_lab.speech.contract import MAX_REQUEST_BYTES, PATH


class Process:
    """A stand-in for the playground's process."""

    def __init__(self, exited=None, stubborn=False):
        self.exited = exited
        self.stubborn = stubborn
        self.calls = []

    def poll(self):
        return self.exited

    def terminate(self):
        self.calls.append("terminate")

    def kill(self):
        self.calls.append("kill")
        self.exited = -9

    def wait(self, timeout=None):
        if self.stubborn and "kill" not in self.calls:
            raise higgs_server.subprocess.TimeoutExpired("playground", timeout)
        return 0


class HiggsServerTests(unittest.TestCase):
    def test_the_handler_is_bound_to_the_worker_and_the_speech_contract(self):
        worker = object()
        handler = higgs_server._host_for(worker)  # type: ignore[arg-type]
        self.assertTrue(issubclass(handler, GenerationHost))
        self.assertEqual((handler.backend, handler.request_path, handler.max_body_bytes,
                          handler.noun), (worker, PATH, MAX_REQUEST_BYTES, "speech"))

    def test_a_playground_that_exits_while_starting_is_an_error(self):
        with self.assertRaisesRegex(RuntimeError, "exited during startup"):
            higgs_server._wait_for_playground(Process(exited=1), 9)  # type: ignore[arg-type]

    def test_a_playground_that_never_answers_times_out(self):
        with mock.patch.object(higgs_server, "PLAYGROUND_START_S", 0.0), \
                self.assertRaises(TimeoutError):
            higgs_server._wait_for_playground(Process(), 9)  # type: ignore[arg-type]

    def test_stopping_asks_first_then_kills(self):
        polite, stubborn = Process(), Process(stubborn=True)
        higgs_server._stop(polite)  # type: ignore[arg-type]
        higgs_server._stop(stubborn)  # type: ignore[arg-type]
        self.assertEqual(polite.calls, ["terminate"])
        self.assertEqual(stubborn.calls, ["terminate", "kill"])

    def test_a_playground_already_gone_is_left_alone(self):
        gone = Process(exited=0)
        higgs_server._stop(gone)  # type: ignore[arg-type]
        self.assertEqual(gone.calls, [])
