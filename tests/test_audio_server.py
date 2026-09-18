import unittest

from ai_lab.audio.server import _torch_device


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


if __name__ == "__main__":
    unittest.main()
