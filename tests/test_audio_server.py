import unittest
from importlib.util import find_spec

from ai_lab.audio.server import _torch_device, _vad_samples


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


if __name__ == "__main__":
    unittest.main()
