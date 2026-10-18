"""Qwen3-TTS behind the speech contract, with a stand-in model."""

import unittest
from threading import Lock
from typing import Any
from unittest import mock

from ai_lab.speech import qwen
from ai_lab.speech.qwen import QwenTtsBackend


class Model:
    def __init__(self):
        self.calls = []

    def generate_voice_design(self, **options):
        self.calls.append(("design", options))
        return [[0.0]], 24000

    def get_supported_speakers(self):
        return ["Vivian", "Ryan"]

    def generate_custom_voice(self, **options):
        self.calls.append(("custom", options))
        return [[0.0]], 24000


def backend(mode: str) -> Any:
    made: Any = QwenTtsBackend.__new__(QwenTtsBackend)
    made.model, made.mode, made.model_name, made.lock = Model(), mode, "qwen", Lock()
    return made


class QwenBackendTests(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(qwen, "wav_result",
                                    lambda model, mode, samples, rate: {"mode": mode})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_voice_design_needs_an_instruction(self):
        with self.assertRaisesRegex(ValueError, "requires a voice instruction"):
            backend("voice-design").generate({"text": "Salut"})

    def test_a_named_voice_is_matched_whatever_its_case(self):
        made = backend("custom-voice")
        made.generate({"text": "Salut", "speaker": "ryan"})
        self.assertEqual(made.model.calls[0][1]["speaker"], "Ryan")

    def test_an_unknown_voice_lists_the_known_ones(self):
        with self.assertRaisesRegex(ValueError, "Vivian, Ryan"):
            backend("custom-voice").generate({"text": "Salut", "speaker": "Nobody"})

    def test_another_model_name_is_refused(self):
        with self.assertRaisesRegex(ValueError, "this engine serves qwen"):
            backend("custom-voice").generate({"text": "Salut", "model": "other"})
