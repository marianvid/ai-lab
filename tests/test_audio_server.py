import unittest
from importlib.util import find_spec

import inspect
import tempfile
from pathlib import Path
from typing import Any

from ai_lab.audio.server import (AudioHost, MlxWhisperBackend, NemoBackend, SileroBackend,
                                 SortformerBackend, _answer, _checkpoint, _language_options,
                                 _torch_device, _vad_samples)


class Availability:
    def __init__(self, available):
        self.available = available

    def is_available(self):
        return self.available


class Torch:
    def __init__(self, cuda=False, mps=False):
        self.cuda = Availability(cuda)
        self.backends = type("Backends", (), {"mps": Availability(mps)})()

    @staticmethod
    def device(name):
        return name


class TorchDeviceTests(unittest.TestCase):
    def test_cuda_is_preferred_when_present(self):
        self.assertEqual(_torch_device(Torch(cuda=True, mps=True)), "cuda")

    def test_mps_is_used_on_apple_silicon(self):
        self.assertEqual(_torch_device(Torch(mps=True)), "mps")

    def test_cpu_is_the_portable_fallback(self):
        self.assertEqual(_torch_device(Torch()), "cpu")


@unittest.skipUnless(find_spec('numpy'), 'NumPy belongs to the isolated VAD runtime')
class VadAudioTests(unittest.TestCase):
    def test_resamples_stereo_recording_to_mono_16khz(self):
        import numpy as np

        stereo = np.column_stack((np.ones(24000, dtype='float32'),
                                  np.zeros(24000, dtype='float32')))
        result = _vad_samples(stereo, 24000)
        self.assertEqual(len(result), 16000)
        self.assertAlmostEqual(float(result.mean()), 0.5)

    def test_rejects_empty_audio(self):
        import numpy as np

        with self.assertRaisesRegex(ValueError, 'empty'):
            _vad_samples(np.array([], dtype='float32'), 16000)



def bare(kind, **answers):
    """A backend of this kind with no model, answering with `answers`."""
    backend = kind.__new__(kind)
    for name, value in answers.items():
        setattr(backend, name, value)
    return backend


class CheckpointTests(unittest.TestCase):
    def test_a_folder_must_hold_exactly_one_nemo_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with self.assertRaisesRegex(RuntimeError, "found 0"):
                _checkpoint(folder)
            (root / "a.nemo").write_bytes(b"")
            self.assertEqual(_checkpoint(folder), str(root / "a.nemo"))
            self.assertEqual(_checkpoint(str(root / "a.nemo")), str(root / "a.nemo"))
            (root / "b.nemo").write_bytes(b"")
            with self.assertRaisesRegex(RuntimeError, "found 2"):
                _checkpoint(folder)


class LanguageTests(unittest.TestCase):
    def test_each_family_is_told_the_language_its_own_way(self):
        self.assertEqual(_language_options("canary", None),
                         {"source_lang": "ro", "target_lang": "ro", "task": "asr", "pnc": "yes"})
        self.assertEqual(_language_options("nemotron", "ro"), {"target_lang": "ro-RO"})
        self.assertEqual(_language_options("nemotron", "de"), {"target_lang": "de"})
        self.assertEqual(_language_options("parakeet", "en"), {"language": "en"})
        self.assertEqual(_language_options("parakeet", None), {})

    def test_only_what_the_model_s_transcribe_takes_is_passed(self):
        def strict(paths, language=None):
            return paths

        def open_ended(paths, **options):
            return paths
        cases: list[tuple[Any, dict]] = [(strict, {"language": "en"}), (open_ended, {})]
        for transcribe, wanted in cases:
            backend = bare(NemoBackend, model_kind="parakeet",
                           model=type("Model", (), {"transcribe": staticmethod(transcribe)})())
            with self.subTest(signature=str(inspect.signature(transcribe))):
                self.assertEqual(backend._options("en"), wanted)


class AnswerTests(unittest.TestCase):
    """What each kind of backend's answer looks like, with no model behind it."""

    def test_transcription_diarization_and_voice_detection(self):
        form: dict[str, tuple[str | None, bytes]] = {"language": (None, b"ro"),
                                                     "threshold": (None, b"0.7")}
        whisper = bare(MlxWhisperBackend, transcribe=lambda _path, language: f"text in {language}")
        self.assertEqual(_answer(whisper, "a.wav", form), {"text": "text in ro"})
        speakers = bare(SortformerBackend, diarize=lambda _path: [
            {"speaker": "B", "start": 1.0}, {"speaker": "A", "start": 0.0}])
        self.assertEqual(_answer(speakers, "a.wav", {})["speakers"], ["A", "B"])
        asked = []

        def segments(_path, threshold, silence):
            asked.append((threshold, silence))
            return []
        vad = bare(SileroBackend, segments=segments)
        self.assertEqual(_answer(vad, "a.wav", form), {"segments": []})
        _answer(vad, "a.wav", {})
        self.assertEqual(asked, [(0.7, 100), (0.5, 100)])

    def test_the_host_keeps_the_recording_s_kind_and_refuses_no_file(self):
        seen = []

        def transcribe(path, _language):
            seen.append(Path(path).suffix)
            return "ok"
        host = AudioHost.__new__(AudioHost)
        host.backend = bare(MlxWhisperBackend, transcribe=transcribe)
        self.assertEqual(host.answer({"file": ("talk.flac", b"x")}), {"text": "ok"})
        self.assertEqual(seen, [".flac"])
        with self.assertRaisesRegex(ValueError, "must contain an audio file"):
            host.answer({})


if __name__ == "__main__":
    unittest.main()
