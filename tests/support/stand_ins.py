"""Stand-ins for the packages the music and speech engines import inside their constructors.

torch, soundfile and each engine's own package (heartlib, mulacover, kokoro,
voxcpm) exist only in the engines' environments, with weights. These let a
backend be built by its real constructor and run its real `generate`, with the
model replaced by something that records what it was asked and writes a WAV
(the reviewer's read of the tests, items 16, 18, 19). What still cannot be
tested this way is the model itself: what the weights make of the request.
"""

from __future__ import annotations

import contextlib
import io
import struct
import sys
import types
import wave
from pathlib import Path
from typing import Any
from unittest import mock


def wav_bytes(seconds: float = 1.0, rate: int = 8000) -> bytes:
    """A silent 16-bit mono WAV of this length."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(rate)
        audio.writeframes(b"\0\0" * int(seconds * rate))
    return buffer.getvalue()


class Tensor:
    """Samples, with the chain of calls the backends make to reach a list."""

    def __init__(self, samples):
        self.samples = list(samples)

    def detach(self):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.samples


def torch(cuda: bool = False) -> types.ModuleType:
    """A torch with no device but the CPU (unless `cuda`), recording every seed."""
    module: Any = types.ModuleType("torch")
    module.seeds = []
    module.bfloat16, module.float32 = "bfloat16", "float32"
    module.device = lambda name: types.SimpleNamespace(type=name.split(":")[0], name=name)
    module.no_grad = module.inference_mode = contextlib.nullcontext
    module.manual_seed = module.seeds.append
    module.cuda = types.SimpleNamespace(is_available=lambda: cuda, manual_seed_all=module.seeds.append)
    module.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False))
    module.cat = lambda chunks: Tensor(sample for chunk in chunks for sample in chunk.samples)
    return module


def soundfile() -> types.ModuleType:
    """soundfile.write for mono float samples, as 16-bit PCM."""
    module: Any = types.ModuleType("soundfile")

    def write(target, samples, rate, format="WAV", subtype="PCM_16"):
        with wave.open(target, "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(rate)
            audio.writeframes(b"".join(struct.pack("<h", int(max(-1.0, min(1.0, value)) * 32767))
                                       for value in samples))
    module.write = write
    return module


class Pipeline:
    """A music pipeline: records how it was built and called, and writes a WAV where asked."""

    built: dict = {}

    def __init__(self, **built):
        self.built = built
        self.calls: list[tuple[dict, dict]] = []
        self.seconds = 2.0

    @classmethod
    def from_pretrained(cls, root, **kwargs):
        return cls(root=root, **kwargs)

    def __call__(self, inputs, **kwargs):
        self.calls.append(({key: Path(value).read_text() if key != "ref_audio"
                            else Path(value).read_bytes() for key, value in inputs.items()}, kwargs))
        Path(kwargs["save_path"]).write_bytes(wav_bytes(self.seconds))


def package(name: str, **members) -> types.ModuleType:
    """A module of this name holding these members."""
    module = types.ModuleType(name)
    for key, value in members.items():
        setattr(module, key, value)
    return module


def installed(**modules: types.ModuleType):
    """Put these stand-ins where `import` finds them, for the length of a `with`."""
    return mock.patch.dict(sys.modules, modules)
